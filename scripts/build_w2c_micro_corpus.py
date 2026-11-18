#!/usr/bin/env python3
"""Builds the 2,000-row Stage 2 v1.1 micro-stage corpus (4-way When2Call MCQ + replay).

Constructs 1,000 balanced 4-way response-strategy MCQ items from the cached or Hub
`nvidia/When2Call` (`train_pref`) split, filtered with a 3-layer zero-overlap filter
against specified evaluation JSONL/JSONL.GZ files, and mixes 1,000 non-overlapping
replay rows from Stage 2/2b training corpora.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import os
import random
import re
import sys
from pathlib import Path

from datasets import load_dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
W2C_INST = "Which response should the assistant give to the user's question, given the available tools?"
CANONICAL_ORDER = ["direct_or_other", "tool_call_json", "ask_clarification", "cannot_answer_refusal"]
_ALNUM_RE = re.compile(r"[^a-z0-9]+")

DEFAULT_REPLAY_QUOTAS = {
    "ClassOne_JevBench_Replay": 250,
    "openai/gsm8k": 100,
    "cais/mmlu_aux": 100,
    "tau/commonsense_qa": 100,
    "allenai/openbookqa": 100,
    "ucinlp/drop": 75,
    "pminervini/HaluEval": 75,
    "nyu-mll/multi_nli": 50,
    "stanfordnlp/snli": 50,
    "mteb/banking77 (train)": 25,
    "clinc/clinc_oos (train)": 25,
    "ContractNLI (tau/scrolls train)": 25,
    "coastalcph/lex_glue_ledgar (train)": 25,
}


def normalize_text(text: str) -> str:
    return _ALNUM_RE.sub(" ", str(text).lower()).strip()


def sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def extract_shingles(norm_txt: str, k_words: int = 7, max_shingles: int = 24) -> list[str]:
    words = norm_txt.split()
    if len(words) < k_words:
        return [norm_txt] if len(norm_txt) >= 30 else []
    total = len(words) - k_words + 1
    step = max(1, total // max_shingles)
    return [" ".join(words[i : i + k_words]) for i in range(0, total, step)][:max_shingles]


class ThreeLayerOverlapFilter:
    """Three-layer overlap detector using state hashes, query-stem hashes, and 7-gram Jaccard."""

    def __init__(self) -> None:
        self.exact_state_hashes: set[str] = set()
        self.question_stem_hashes: set[str] = set()
        self.shingle_to_doc_ids: dict[str, set[int]] = collections.defaultdict(set)
        self.doc_shingle_counts: dict[int, int] = {}
        self._next_doc_id = 0

    def add_forbidden_text(self, state_text: str, question_texts: list[str] | None = None) -> None:
        norm_s = normalize_text(state_text)
        if norm_s:
            self.exact_state_hashes.add(sha256_str(norm_s))
            self.exact_state_hashes.add(sha256_str(norm_s[:250]))
            shingles = extract_shingles(norm_s)
            if shingles:
                doc_id = self._next_doc_id
                self._next_doc_id += 1
                self.doc_shingle_counts[doc_id] = len(shingles)
                for sh in shingles:
                    if len(self.shingle_to_doc_ids[sh]) < 50:
                        self.shingle_to_doc_ids[sh].add(doc_id)

        if question_texts:
            for q_txt in question_texts:
                norm_q = normalize_text(q_txt)
                if len(norm_q) >= 25:
                    self.question_stem_hashes.add(sha256_str(norm_q))
                    self.question_stem_hashes.add(sha256_str(norm_q[:180]))

    def check_overlap(
        self,
        state_text: str,
        question_texts: list[str] | None = None,
        jaccard_thresh: float = 0.45,
    ) -> str | None:
        norm_s = normalize_text(state_text)
        if norm_s:
            if sha256_str(norm_s) in self.exact_state_hashes or sha256_str(norm_s[:250]) in self.exact_state_hashes:
                return "layer1_exact_hash"

        if question_texts:
            for q_txt in question_texts:
                norm_q = normalize_text(q_txt)
                if len(norm_q) >= 25 and (
                    sha256_str(norm_q) in self.question_stem_hashes
                    or sha256_str(norm_q[:180]) in self.question_stem_hashes
                ):
                    return "layer2_question_stem"

        if norm_s:
            shingles = extract_shingles(norm_s)
            if len(shingles) >= 3:
                hit_counts: dict[int, int] = collections.defaultdict(int)
                for sh in shingles:
                    for doc_id in self.shingle_to_doc_ids.get(sh, ()):
                        hit_counts[doc_id] += 1
                for doc_id, hits in hit_counts.items():
                    other_len = self.doc_shingle_counts.get(doc_id, len(shingles))
                    union = len(shingles) + other_len - hits
                    if (hits / max(1, union)) >= jaccard_thresh or (hits / len(shingles)) >= 0.60:
                        return "layer3_ngram_jaccard"
        return None


def index_eval_paths(eval_paths: list[Path]) -> ThreeLayerOverlapFilter:
    """Indexes evaluation datasets by normalized state/query text without blocking generic instructions."""
    flt = ThreeLayerOverlapFilter()
    indexed = 0
    for path in eval_paths:
        if not path.exists():
            continue
        files = list(path.rglob("*.jsonl*")) if path.is_dir() else [path]
        for file_path in files:
            opener = gzip.open if file_path.name.endswith(".gz") else open
            with opener(file_path, "rt", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    indexed += 1
                    row = json.loads(line)
                    st = row.get("state", row.get("prompt", ""))
                    s_txt = st if isinstance(st, str) else json.dumps(st, ensure_ascii=False, sort_keys=True)
                    q_txts: list[str] = []
                    if isinstance(st, dict) and "question" in st and "tools" in st:
                        uq = str(st.get("question", "")).strip()
                        if uq:
                            q_txts.append(uq)
                            norm_uq = normalize_text(uq)
                            if norm_uq:
                                flt.exact_state_hashes.add(sha256_str(norm_uq))
                    else:
                        for q in row.get("questions", {}).values():
                            if isinstance(q, dict):
                                inst = q.get("instructions", "")
                                inst_s = inst if isinstance(inst, str) else json.dumps(inst, ensure_ascii=False)
                                if inst_s.strip() != W2C_INST:
                                    q_txts.append(inst_s)
                        if len(s_txt) < 400:
                            q_txts.append(s_txt)
                    flt.add_forbidden_text(s_txt, q_txts)
    print(f"[Filter] Indexed {indexed} eval rows; {len(flt.exact_state_hashes)} hashes.", flush=True)
    return flt


def parse_toolcall_and_classify(resp_obj: dict | str) -> tuple[str, str]:
    """Classifies a response into the 4 canonical strategies and normalizes tool-call JSON."""
    c = resp_obj.get("content", "") if isinstance(resp_obj, dict) else str(resp_obj)
    s = str(c).strip()
    cl = s.lower()
    if "<toolcall>" in cl:
        m = re.search(r"<toolcall>(.*?)</toolcall>", s, re.DOTALL | re.IGNORECASE)
        raw_inner = m.group(1).strip() if m else s
        try:
            parsed = json.loads(raw_inner)
            if isinstance(parsed, list) and parsed:
                parsed = parsed[0]
            if isinstance(parsed, dict) and "name" in parsed:
                norm = json.dumps(
                    {"name": parsed["name"], "arguments": parsed.get("arguments", {})}, ensure_ascii=False
                )
                return "tool_call_json", norm
        except Exception:
            pass
    if s.startswith("{") and "name" in s and "arguments" in s:
        return "tool_call_json", s
    if any(
        p in cl
        for p in (
            "unable to",
            "cannot ",
            "apologies",
            "i apologize",
            "no tool",
            "beyond my capabilities",
            "not have access",
            "not able to",
            "can not",
            "sorry",
        )
    ):
        return "cannot_answer_refusal", s
    if "?" in s:
        return "ask_clarification", s
    return "direct_or_other", s


def synth_tool_call_for_prompt(tools: list[dict], rng: random.Random) -> str:
    """Synthesizes a tool-call JSON option matching the available tools schema for a prompt."""
    if not tools:
        return json.dumps({"name": "search", "arguments": {"query": "details"}}, ensure_ascii=False)
    t = rng.choice(tools)
    props = ((t.get("parameters") or {}).get("properties") or {}) if isinstance(t, dict) else {}
    req = ((t.get("parameters") or {}).get("required") or []) if isinstance(t, dict) else []
    args = {}
    keys = list(req) if req else list(props.keys())[:2]
    for k in keys[:3]:
        pinfo = props.get(k, {}) if isinstance(props.get(k), dict) else {}
        ptype = str(pinfo.get("type", "string")).lower()
        enum_vals = pinfo.get("enum")
        if isinstance(enum_vals, list) and enum_vals:
            args[k] = enum_vals[0]
        elif ptype in ("integer", "int"):
            args[k] = 1
        elif ptype in ("number", "float"):
            args[k] = 1.0
        elif ptype in ("boolean", "bool"):
            args[k] = True
        elif ptype == "array":
            args[k] = ["item"]
        elif ptype in ("object", "dict"):
            args[k] = {}
        else:
            args[k] = "example"
    return json.dumps({"name": t.get("name", "tool"), "arguments": args}, ensure_ascii=False)


def build_w2c_micro_corpus(
    *,
    stage2b_corpus: Path | None,
    combined_corpus: Path | None,
    output_path: Path,
    eval_paths: list[Path],
    seed: int = 20261118,
) -> dict:
    """Constructs the 2,000-row 4-way When2Call + replay micro-stage corpus."""
    rng = random.Random(seed)
    flt = index_eval_paths(eval_paths)

    ds = load_dataset("nvidia/When2Call", "train_pref")["train"]
    pools: dict[str, list[str]] = collections.defaultdict(list)
    by_prompt: dict[tuple[str, tuple[str, ...]], list[dict]] = collections.defaultdict(list)

    overlap_blocked = 0
    for row in ds:
        user_q = ""
        for m in row.get("messages") or []:
            if isinstance(m, dict) and m.get("role") == "user":
                user_q = str(m.get("content", "")).strip()
        if not user_q:
            continue
        raw_tools = row.get("tools") or []
        tools = []
        for t in raw_tools:
            try:
                tools.append(json.loads(t) if isinstance(t, str) else t)
            except Exception:
                pass
        state = {"tools": tools, "question": user_q}
        state_str = json.dumps(state, ensure_ascii=False, sort_keys=True)
        norm_q = normalize_text(user_q)
        if flt.check_overlap(state_str, [user_q]) is not None or sha256_str(norm_q) in flt.exact_state_hashes:
            overlap_blocked += 1
            continue

        ct, cs = parse_toolcall_and_classify(row["chosen_response"])
        rt, rs = parse_toolcall_and_classify(row["rejected_response"])
        pools[ct].append(cs)
        pools[rt].append(rs)
        by_prompt[(user_q, tuple(str(t) for t in raw_tools))].append(
            {"state": state, "chosen_type": ct, "chosen_text": cs, "rejected_type": rt, "rejected_text": rs}
        )

    for k in pools:
        pools[k] = sorted(set(pools[k]))
        rng.shuffle(pools[k])

    by_gold: dict[str, list[tuple[tuple, list[dict]]]] = collections.defaultdict(list)
    for key, entries in sorted(by_prompt.items()):
        by_gold[entries[0]["chosen_type"]].append((key, entries))
    for gtype in by_gold:
        rng.shuffle(by_gold[gtype])

    n_direct = min(36, len(by_gold["direct_or_other"]))
    quotas = {
        "direct_or_other": n_direct,
        "cannot_answer_refusal": 322,
        "ask_clarification": 322,
        "tool_call_json": 320 + (36 - n_direct),
    }

    w2c_rows: list[dict] = []
    pool_idx = {k: 0 for k in CANONICAL_ORDER}

    def _next_pool_text(cat: str, avoid: set[str]) -> str:
        arr = pools[cat]
        for _ in range(len(arr)):
            val = arr[pool_idx[cat] % len(arr)]
            pool_idx[cat] += 1
            if val not in avoid:
                return val
        return arr[0]

    row_seq = 0
    for gtype, need in quotas.items():
        selected = by_gold[gtype][:need]
        assert len(selected) == need, f"Insufficient {gtype} prompts: {len(selected)} < {need}"
        for _, entries in selected:
            e0 = entries[0]
            state = e0["state"]
            opts_by_cat: dict[str, str] = {e0["chosen_type"]: e0["chosen_text"]}
            for ent in entries:
                rt, rs = ent["rejected_type"], ent["rejected_text"]
                if rt not in opts_by_cat and rs != e0["chosen_text"]:
                    opts_by_cat[rt] = rs
            used_texts = set(opts_by_cat.values())
            for cat in CANONICAL_ORDER:
                if cat in opts_by_cat:
                    continue
                txt = (
                    synth_tool_call_for_prompt(state["tools"], rng)
                    if cat == "tool_call_json"
                    else _next_pool_text(cat, used_texts)
                )
                opts_by_cat[cat] = txt
                used_texts.add(txt)

            order = list(CANONICAL_ORDER)
            if row_seq % 2 != 0:
                rng.shuffle(order)

            letters = ["A", "B", "C", "D"]
            criteria = {letters[i]: opts_by_cat[order[i]] for i in range(4)}
            gold_letter = letters[order.index(e0["chosen_type"])]
            rid = hashlib.sha256(f"{state['question']}:{row_seq}".encode()).hexdigest()[:12]
            w2c_rows.append(
                {
                    "id": f"w2c_4way_{rid}",
                    "source": "nvidia/When2Call_4way_mcq (train_pref)",
                    "state": state,
                    "questions": {"q": {"type": "choice", "instructions": W2C_INST, "criteria": criteria}},
                    "targets": {"q": gold_letter},
                    "_gold_strategy": e0["chosen_type"],
                    "_canonical_layout": (row_seq % 2 == 0),
                }
            )
            row_seq += 1

    replay_rows: list[dict] = []
    replay_sources = [p for p in (stage2b_corpus, combined_corpus) if p is not None and p.exists()]
    if replay_sources:
        replay_by_src: dict[str, list[dict]] = collections.defaultdict(list)
        seen_ids: set[str] = set()
        for src_file in replay_sources:
            for line in src_file.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                src_name = r.get("source", "")
                rid = str(r.get("id", ""))
                if src_name in DEFAULT_REPLAY_QUOTAS and rid not in seen_ids:
                    seen_ids.add(rid)
                    replay_by_src[src_name].append(r)

        for src_name, qn in DEFAULT_REPLAY_QUOTAS.items():
            cand = replay_by_src[src_name]
            rng.shuffle(cand)
            assert len(cand) >= qn, f"Insufficient replay rows for {src_name}: {len(cand)} < {qn}"
            replay_rows.extend(cand[:qn])

    all_rows = w2c_rows + replay_rows
    rng.shuffle(all_rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in all_rows) + "\n", encoding="utf-8")

    return {
        "total_rows": len(all_rows),
        "when2call_4way_rows": len(w2c_rows),
        "replay_rows": len(replay_rows),
        "overlap_blocked_rows": overlap_blocked,
        "gold_slot_dist": dict(sorted(collections.Counter(r["targets"]["q"] for r in w2c_rows).items())),
        "gold_strategy_dist": dict(collections.Counter(r["_gold_strategy"] for r in w2c_rows)),
    }


def main() -> None:
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Build 2,000-row 4-way When2Call + replay micro-stage corpus")
    parser.add_argument("--stage2b-corpus", type=Path, default=None, help="Path to Stage 2b corpus JSONL for replay")
    parser.add_argument("--combined-corpus", type=Path, default=None, help="Path to Stage 2 combined corpus JSONL")
    parser.add_argument("--output-path", type=Path, required=True, help="Output JSONL path")
    parser.add_argument(
        "--eval-paths", type=Path, nargs="*", default=[], help="Evaluation JSONL/dir paths for 3-layer filtering"
    )
    parser.add_argument("--seed", type=int, default=20261118)
    parser.add_argument("--offline", action="store_true", help="Force HuggingFace datasets offline mode")
    args = parser.parse_args()

    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    default_evals = list((REPO_ROOT / "data" / "jevbench").glob("*.jsonl")) + list(
        (REPO_ROOT / "data" / "RLCDAlignBench").glob("*.jsonl")
    )
    summary = build_w2c_micro_corpus(
        stage2b_corpus=args.stage2b_corpus,
        combined_corpus=args.combined_corpus,
        output_path=args.output_path,
        eval_paths=list(args.eval_paths) + default_evals,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

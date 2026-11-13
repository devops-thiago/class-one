"""Tests preventing benchmark attribution drift, stale claims, and regressed Hub uploads."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from classone.publishing import (
    check_candidate_regression,
    compute_sha256,
    find_forbidden_phrases,
    load_baseline_registry,
    parse_readme_heads_hashes,
    validate_markdown_attribution,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PUSH_SCRIPT = REPO_ROOT / "scripts" / "push_to_hub.py"
VERIFY_HUB_SCRIPT = REPO_ROOT / "scripts" / "verify_hub_heads_sha.py"


def _load_push_module():
    spec = importlib.util.spec_from_file_location("push_to_hub_mod", PUSH_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_readme_benchmark_attribution_and_registry_sync():
    readme_text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    errors = validate_markdown_attribution(
        readme_text,
        source_name="README.md",
        require_official_section=True,
    )
    assert errors == []

    registry = load_baseline_registry()
    for repo_id, entry in registry["models"].items():
        assert repo_id in readme_text, f"{repo_id} missing from README.md"
        assert entry["heads_sha256"] in readme_text, f"SHA256 for {repo_id} missing from README.md"


def test_docs_and_skills_contain_no_forbidden_claims():
    targets = list((REPO_ROOT / "docs").glob("*.md")) + list((REPO_ROOT / ".agents" / "skills").rglob("*.md"))
    assert targets, "Expected docs/ and .agents/skills/ markdown files"
    for md_path in targets:
        text = md_path.read_text(encoding="utf-8")
        bad = find_forbidden_phrases(text)
        assert bad == [], f"{md_path} contains forbidden phrases: {bad}"


@pytest.mark.parametrize("standalone", [True, False])
def test_generated_model_cards_for_all_registered_models(standalone: bool):
    push_mod = _load_push_module()
    registry = load_baseline_registry()
    for repo_id, entry in registry["models"].items():
        card = push_mod.generate_model_card(
            repo_id=repo_id,
            base_model=entry["base_model"],
            standalone=standalone,
            commit_sha="deadbeef",
        )
        errors = validate_markdown_attribution(
            card,
            source_name=f"{repo_id} (standalone={standalone})",
            require_official_section=True,
        )
        assert errors == []
        assert "deadbeef" in card
        assert entry["heads_sha256"] in card, f"Full SHA256 missing from generated model card for {repo_id}"

        lora_card = push_mod.generate_lora_card(repo_id=repo_id, base_model=entry["base_model"])
        lora_errors = validate_markdown_attribution(
            lora_card,
            source_name=f"{repo_id} lora",
            require_official_section=False,
        )
        assert lora_errors == []


def test_validator_catches_missing_table_provenance_and_forbidden_phrases():
    bad_card = """
# ClassOne Demo
This is #1 on Benchmark Heaven Leaderboard on the HF Decision Index.

| Tier | Tasks | Accuracy |
|---|---|---|
| Easy | 48 | 100.0% |
"""
    errors = validate_markdown_attribution(bad_card, source_name="bad_card", require_official_section=True)
    joined = "\n".join(errors)
    assert "#1 on Benchmark Heaven" in joined
    assert "HF Decision Index" in joined
    assert "missing self-measurement metadata" in joined
    assert "Benchmark Heaven leaderboard URL" in joined


def test_regression_gate_detects_dropped_metrics_and_sha_mismatch(tmp_path: Path):
    heads_file = tmp_path / "classone_heads.pt"
    heads_file.write_bytes(b"candidate-head-bytes")
    actual_sha = compute_sha256(heads_file)

    passing_report = {
        "heads_sha256": actual_sha,
        "jevbench_public_acc": 0.71,
        "rlcd_alignbench_bal_acc": 0.58,
    }
    assert (
        check_candidate_regression(
            "devops-thiago/classone-gemma4-e2b",
            passing_report,
            heads_path=heads_file,
        )
        == []
    )

    regressed_report = {
        "heads_sha256": actual_sha,
        "jevbench_public_acc": 0.584,
        "rlcd_alignbench_bal_acc": 0.501,
    }
    failures = check_candidate_regression(
        "devops-thiago/classone-gemma4-e2b",
        regressed_report,
        heads_path=heads_file,
    )
    assert len(failures) == 2
    assert "jevbench_public_acc" in failures[0]

    mismatched_sha_report = {
        "heads_sha256": "0" * 64,
        "jevbench_public_acc": 0.71,
        "rlcd_alignbench_bal_acc": 0.58,
    }
    sha_failures = check_candidate_regression(
        "devops-thiago/classone-gemma4-e2b",
        mismatched_sha_report,
        heads_path=heads_file,
    )
    assert any("does not match checkpoint" in msg for msg in sha_failures)


def test_push_to_hub_cli_gate_blocks_missing_report_and_regression(tmp_path: Path):
    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()
    heads_file = ckpt_dir / "classone_heads.pt"
    heads_file.write_bytes(b"candidate-head-bytes")

    # 1. Missing --eval-report must fail before touching network or token.
    res_missing = subprocess.run(
        [
            sys.executable,
            str(PUSH_SCRIPT),
            "--checkpoint-dir",
            str(ckpt_dir),
            "--repo-id",
            "devops-thiago/classone-gemma4-e2b",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res_missing.returncode == 1
    assert "Upload blocked" in res_missing.stdout

    # 2. Regressed --eval-report must fail with regression details.
    report_path = tmp_path / "eval_report.json"
    report_path.write_text(
        json.dumps(
            {
                "heads_sha256": compute_sha256(heads_file),
                "jevbench_public_acc": 0.584,
                "rlcd_alignbench_bal_acc": 0.510,
            }
        ),
        encoding="utf-8",
    )
    res_regressed = subprocess.run(
        [
            sys.executable,
            str(PUSH_SCRIPT),
            "--checkpoint-dir",
            str(ckpt_dir),
            "--repo-id",
            "devops-thiago/classone-gemma4-e2b",
            "--eval-report",
            str(report_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res_regressed.returncode == 1
    assert "Regression gate blocked upload" in res_regressed.stdout


def test_parse_readme_hashes_and_hub_verifier_offline():
    readme_text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    parsed = parse_readme_heads_hashes(readme_text)
    registry = load_baseline_registry()

    assert set(parsed.keys()) == set(registry["models"].keys())
    for repo_id, entry in registry["models"].items():
        assert parsed[repo_id] == entry["heads_sha256"].upper()

    spec = importlib.util.spec_from_file_location("verify_hub_heads_sha_mod", VERIFY_HUB_SCRIPT)
    assert spec is not None and spec.loader is not None
    verify_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verify_mod)

    class _FakeLfs:
        def __init__(self, sha: str):
            self.sha256 = sha

    class _FakePathInfo:
        def __init__(self, sha: str, size: int = 1234):
            self.lfs = _FakeLfs(sha)
            self.size = size

    class _MatchingApi:
        def get_paths_info(self, repo_id: str, paths: list[str]):
            assert paths == ["classone_heads.pt"]
            return [_FakePathInfo(parsed[repo_id].lower())]

    assert verify_mod.verify_hub_heads(api=_MatchingApi()) == []

    class _MismatchedApi:
        def get_paths_info(self, repo_id: str, paths: list[str]):
            sha = "0" * 64 if "e2b" in repo_id else parsed[repo_id]
            return [_FakePathInfo(sha)]

    mismatches = verify_mod.verify_hub_heads(api=_MismatchedApi())
    assert len(mismatches) == 1
    assert "classone-gemma4-e2b" in mismatches[0]

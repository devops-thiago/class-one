"""Publishing validation utilities for model cards, docs, and non-regression gates."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

FORBIDDEN_CARD_PHRASES: tuple[str, ...] = (
    "Leaderboard Rank #1",
    "#1 on Benchmark Heaven",
    "HF Decision Index",
)

BENCHMARK_HEADER_MARKERS: tuple[str, ...] = (
    "Accuracy",
    "AUROC",
    "Score (0–100)",
    "Score (0-100)",
    "Domain Score",
    "Composite Score",
    "Easy Tier",
    "Overall (231)",
    "Latency (p50)",
    "Median Latency",
)

DEFAULT_BASELINES_PATH = Path(__file__).resolve().parents[2] / "benchmarks" / "baselines.json"


def compute_sha256(file_path: str | Path) -> str:
    """Computes uppercase hex SHA256 digest of a file."""
    digest = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def load_baseline_registry(path: str | Path | None = None) -> dict[str, Any]:
    """Loads the canonical benchmark baseline registry from disk."""
    registry_path = Path(path) if path is not None else DEFAULT_BASELINES_PATH
    with open(registry_path, encoding="utf-8") as f:
        return json.load(f)


def find_forbidden_phrases(text: str) -> list[str]:
    """Returns any disallowed leaderboard or misleading index phrases present in text."""
    lowered = text.lower()
    return [phrase for phrase in FORBIDDEN_CARD_PHRASES if phrase.lower() in lowered]


def validate_markdown_attribution(
    text: str,
    *,
    source_name: str = "document",
    require_official_section: bool = False,
) -> list[str]:
    """Validates benchmark attribution metadata and official-result alignment in markdown."""
    errors: list[str] = []

    for phrase in find_forbidden_phrases(text):
        errors.append(f"{source_name}: forbidden phrase detected: {phrase!r}")

    lines = text.splitlines()
    for idx, line in enumerate(lines):
        stripped = line.strip()
        is_table_header = (
            stripped.startswith("|")
            and stripped.endswith("|")
            and idx + 1 < len(lines)
            and re.match(r"^\|[\s:\-|]+\|$", lines[idx + 1].strip()) is not None
        )
        if not is_table_header:
            continue

        if not any(marker.lower() in stripped.lower() for marker in BENCHMARK_HEADER_MARKERS):
            continue

        context_window = "\n".join(lines[max(0, idx - 6) : idx])
        ctx_lower = context_window.lower()

        # Official Benchmark Heaven tables cite the external source directly.
        is_official_table = "benchmark heaven" in stripped.lower() or (
            "official" in ctx_lower and "benchmarkheaven.com/jev-models" in ctx_lower
        )
        if is_official_table:
            continue

        missing = [token for token in ("RTX 5060 Ti", "Date:", "Commit:") if token.lower() not in ctx_lower]
        if missing:
            errors.append(
                f"{source_name}: line {idx + 1} benchmark table missing self-measurement metadata ({', '.join(missing)})"
            )

    if require_official_section:
        lowered = text.lower()
        if "benchmark heaven" not in lowered:
            errors.append(f"{source_name}: missing Benchmark Heaven official results section")
        if "https://benchmarkheaven.com/jev-models" not in text:
            errors.append(f"{source_name}: missing Benchmark Heaven leaderboard URL")
        if "sealed" not in lowered:
            errors.append(f"{source_name}: missing explanation of sealed vs. public differences")

    return errors


README_HASH_ROW_RE = re.compile(
    r"^\|\s*\[`([^`]+)`\]\([^)]+\)\s*\|[^|]*\|[^|]*\|\s*`([0-9A-Fa-f]{64})`\s*\|",
)


def parse_readme_heads_hashes(readme_text: str) -> dict[str, str]:
    """Parses the 'Calibrated Heads SHA256 (Verified on Hub)' table from README.md."""
    parsed: dict[str, str] = {}
    for line in readme_text.splitlines():
        match = README_HASH_ROW_RE.match(line.strip())
        if match:
            repo_id, sha256_hex = match.groups()
            parsed[repo_id.strip()] = sha256_hex.upper()
    return parsed


def get_registered_heads_sha256(repo_id: str, registry: dict[str, Any] | None = None) -> str | None:
    """Returns the canonical uppercase SHA256 digest for repo_id from baselines.json."""
    try:
        data = registry if registry is not None else load_baseline_registry()
    except OSError:
        return None
    sha = data.get("models", {}).get(repo_id, {}).get("heads_sha256")
    return str(sha).strip().upper() if sha else None


def _normalize_rate(value: float, metric_name: str) -> float:
    """Normalizes percentage inputs (0..100) for fractional metrics when needed."""
    if metric_name in {"jevbench_public_acc", "rlcd_alignbench_bal_acc"} and value > 1.0:
        return value / 100.0
    return value


def check_candidate_regression(
    repo_id: str,
    eval_report: dict[str, Any],
    *,
    heads_path: str | Path | None = None,
    registry: dict[str, Any] | None = None,
    tolerance: float = 1e-4,
) -> list[str]:
    """Validates candidate metrics and head SHA256 against checked-in baseline floors."""
    errors: list[str] = []
    data = registry if registry is not None else load_baseline_registry()
    model_entry = data.get("models", {}).get(repo_id)

    if model_entry is None:
        return errors

    if heads_path is not None and "heads_sha256" in eval_report:
        actual_sha = compute_sha256(heads_path)
        expected_sha = str(eval_report["heads_sha256"]).strip().upper()
        if actual_sha != expected_sha:
            errors.append(
                f"Eval report heads_sha256 ({expected_sha[:8]}...) does not match checkpoint ({actual_sha[:8]}...)"
            )

    metrics_map = eval_report.get("metrics", eval_report)
    for metric_name, floor in model_entry.get("floors", {}).items():
        if metric_name not in metrics_map or metrics_map[metric_name] is None:
            errors.append(f"{repo_id}: eval report missing required baseline metric {metric_name!r}")
            continue
        candidate_val = _normalize_rate(float(metrics_map[metric_name]), metric_name)
        if candidate_val + tolerance < float(floor):
            errors.append(
                f"{repo_id}: metric {metric_name!r} regressed ({candidate_val:.4f} < baseline floor {float(floor):.4f})"
            )

    return errors

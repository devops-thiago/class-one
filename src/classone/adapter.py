"""ClassOne Adapter module for JevBench offline evaluation harness."""

import os
import sys

try:
    from classone_adapter import ClassOneAdapter, DecisionResult
except ImportError:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from classone_adapter import ClassOneAdapter, DecisionResult

__all__ = ["ClassOneAdapter", "DecisionResult"]

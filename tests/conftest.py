"""PyTorch CUDA memory hygiene fixtures for ClassOne test suite."""

import gc

import pytest
import torch


@pytest.fixture(autouse=True)
def clean_cuda_memory():
    """Ensures GPU memory cache is emptied and garbage collected between tests."""
    yield
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

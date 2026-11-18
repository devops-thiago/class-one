"""Configuration for ClassOne System 1 decision models."""

from __future__ import annotations

from typing import Any

from transformers import PretrainedConfig


class ClassOneConfig(PretrainedConfig):
    """Configuration class for ClassOne decision model."""

    model_type = "classone"

    def __init__(
        self,
        base_model_name_or_path: str = "google/gemma-2-2b-it",
        hidden_size: int = 2048,
        head_hidden_size: int = 512,
        dropout: float = 0.1,
        temperature: float = 1.0,
        use_temperature_scaling: bool = True,
        max_position_embeddings: int | None = None,
        eval_mode: str = "stage1",
        m2_k_threshold: int = 64,
        m2_top_k: int = 16,
        classone_config: dict[str, Any] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.base_model_name_or_path = base_model_name_or_path
        self.hidden_size = hidden_size
        self.head_hidden_size = head_hidden_size
        self.dropout = dropout
        self.temperature = temperature
        self.use_temperature_scaling = use_temperature_scaling
        self.max_position_embeddings = max_position_embeddings
        self.eval_mode = eval_mode
        self.m2_k_threshold = m2_k_threshold
        self.m2_top_k = m2_top_k
        self.classone_config = classone_config

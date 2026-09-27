"""Configuration for ClassOne System 1 decision models."""

from __future__ import annotations

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
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.base_model_name_or_path = base_model_name_or_path
        self.hidden_size = hidden_size
        self.head_hidden_size = head_hidden_size
        self.dropout = dropout
        self.temperature = temperature
        self.use_temperature_scaling = use_temperature_scaling

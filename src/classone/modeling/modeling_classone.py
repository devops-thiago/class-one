"""Model architecture for ClassOne System One decision models.

Integrates a transformer backbone (e.g. Gemma 4 E2B) with lightweight decision heads
for parallel single-pass evaluation of Noul, Choice, and Score primitives.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel, PreTrainedModel

from classone.modeling.configuration_classone import ClassOneConfig
from classone.schemas import ChoiceResult, NoulResult, Result, ScoreResult
from classone.tokenizer import PackedSequence


class NoulHead(nn.Module):
    """Boolean decision head predicting calibrated probability P(true)."""

    def __init__(self, hidden_size: int, head_hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, head_hidden_size),
            nn.LayerNorm(head_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_size, 1),
        )
        # Smooth strictly positive temperature parameterization: T = softplus(raw) + 0.1
        self.temperature_raw = nn.Parameter(torch.zeros(1))

    @property
    def temperature(self) -> torch.Tensor:
        return F.softplus(self.temperature_raw) + 0.1

    def forward(self, h_query: torch.Tensor) -> torch.Tensor:
        """Args:

        h_query: [batch_size, hidden_size] or [hidden_size]
        Returns:
            probability tensor in [0, 1]
        """
        if h_query.dim() == 1:
            h_query = h_query.unsqueeze(0)
        logit = self.net(h_query).squeeze(-1)
        prob = torch.sigmoid(logit / self.temperature)
        return prob


class ChoiceHead(nn.Module):
    """Categorical decision head computing calibrated distribution over dynamic options."""

    def __init__(self, hidden_size: int, head_hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.q_proj = nn.Sequential(
            nn.Linear(hidden_size, head_hidden_size),
            nn.LayerNorm(head_hidden_size),
            nn.Dropout(dropout),
        )
        self.k_proj = nn.Sequential(
            nn.Linear(hidden_size, head_hidden_size),
            nn.LayerNorm(head_hidden_size),
            nn.Dropout(dropout),
        )
        self.scale = 1.0 / math.sqrt(head_hidden_size)
        self.temperature_raw = nn.Parameter(torch.zeros(1))

    @property
    def temperature(self) -> torch.Tensor:
        return F.softplus(self.temperature_raw) + 0.1

    def forward(self, h_query: torch.Tensor, h_options: torch.Tensor) -> torch.Tensor:
        """Args:

        h_query: Query token vector [hidden_size]
        h_options: Candidate option token vectors [num_options, hidden_size]
        Returns:
            Probability distribution over options [num_options]
        """
        if h_query.dim() == 1:
            h_query = h_query.unsqueeze(0)  # [1, hidden_size]

        q = self.q_proj(h_query)  # [1, head_hidden_size]
        k = self.k_proj(h_options)  # [num_options, head_hidden_size]

        # Dot-product similarity
        logits = torch.matmul(q, k.transpose(0, 1)).squeeze(0) * self.scale
        probs = F.softmax(logits / self.temperature, dim=-1)
        return probs


class ScoreHead(nn.Module):
    """Ordinal decision head predicting expected score over an ordered rubric."""

    def __init__(self, hidden_size: int, head_hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.choice_evaluator = ChoiceHead(hidden_size, head_hidden_size, dropout)

    @property
    def temperature(self) -> torch.Tensor:
        return self.choice_evaluator.temperature

    def forward(self, h_query: torch.Tensor, h_levels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Args:

        h_query: Query token vector [hidden_size]
        h_levels: Rubric level vectors [num_levels, hidden_size]
        Returns:
            (expected_score, probabilities)
        """
        probs = self.choice_evaluator(h_query, h_levels)
        num_levels = probs.shape[-1]
        # Match level_weights dtype to probs dtype to prevent Half/Float runtime crash
        level_weights = torch.arange(1, num_levels + 1, dtype=probs.dtype, device=probs.device)
        expected_score = torch.sum(probs * level_weights)
        return expected_score, probs


class ClassOneModel(PreTrainedModel):
    """Complete ClassOne decision model wrapping a transformer backbone with System 1 heads."""

    config_class = ClassOneConfig

    def __init__(self, config: ClassOneConfig, backbone: nn.Module | None = None):
        super().__init__(config)
        self.config = config

        if backbone is not None:
            self.backbone = backbone
        else:
            # Standalone lightweight transformer encoder for local testing/CPU
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=config.hidden_size,
                nhead=8,
                dim_feedforward=config.hidden_size * 2,
                dropout=config.dropout,
                batch_first=True,
            )
            self.backbone = nn.TransformerEncoder(encoder_layer, num_layers=4, enable_nested_tensor=False)
            self.embed = nn.Embedding(32000, config.hidden_size)

        # Decision Heads
        self.noul_head = NoulHead(config.hidden_size, config.head_hidden_size, config.dropout)
        self.choice_head = ChoiceHead(config.hidden_size, config.head_hidden_size, config.dropout)
        self.score_head = ScoreHead(config.hidden_size, config.head_hidden_size, config.dropout)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Standard PreTrainedModel forward pass extracting hidden states."""
        return self.extract_hidden_states(input_ids=input_ids, attention_mask=attention_mask)

    @property
    def device(self) -> torch.device:
        """Returns the device of the model parameters."""
        return next(self.parameters()).device

    @classmethod
    def from_backbone(
        cls,
        base_model_name_or_path: str | nn.Module,
        config: ClassOneConfig | None = None,
        tokenizer: Any | None = None,
        torch_dtype: torch.dtype | None = None,
        device: str | torch.device | None = None,
        **backbone_kwargs,
    ) -> ClassOneModel:
        """Instantiates ClassOne with an existing HuggingFace backbone (e.g. Gemma 4 E2B).

        Args:
            base_model_name_or_path: HuggingFace model identifier, local directory, or existing nn.Module.
            config: Optional ClassOneConfig override.
            tokenizer: Optional tokenizer; if provided, backbone token embeddings are resized.
            torch_dtype: Precision dtype (e.g. torch.bfloat16, torch.float16, torch.float32).
            device: Target device ('mps', 'cuda', 'cpu'); if None, detects automatically.
            **backbone_kwargs: Extra keyword arguments forwarded to AutoModel.from_pretrained.
        """
        if isinstance(base_model_name_or_path, nn.Module):
            backbone = base_model_name_or_path
            model_name = getattr(backbone, "name_or_path", "custom_module")
        else:
            model_name = str(base_model_name_or_path)
            kwargs = dict(backbone_kwargs)
            if torch_dtype is not None:
                kwargs["torch_dtype"] = torch_dtype
            backbone = AutoModel.from_pretrained(model_name, **kwargs)

        # Resize token embeddings if tokenizer includes ClassOne delimiter tokens
        if tokenizer is not None and hasattr(backbone, "resize_token_embeddings"):
            try:
                backbone.resize_token_embeddings(len(tokenizer), mean_resizing=False)
            except TypeError:
                backbone.resize_token_embeddings(len(tokenizer))

        text_config = getattr(backbone.config, "text_config", None)
        hidden_size = (
            getattr(text_config, "hidden_size", None)
            or getattr(backbone.config, "hidden_size", None)
            or getattr(backbone.config, "d_model", None)
            or 2048
        )
        if config is None:
            config = ClassOneConfig(
                base_model_name_or_path=model_name,
                hidden_size=hidden_size,
            )
        else:
            config.hidden_size = hidden_size

        model = cls(config=config, backbone=backbone)

        # Determine target device
        if device is None:
            if torch.cuda.is_available():
                target_device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                target_device = torch.device("mps")
            else:
                target_device = torch.device("cpu")
        else:
            target_device = torch.device(device)

        model.to(target_device)
        if torch_dtype is not None:
            model.to(dtype=torch_dtype)

        return model

    def extract_hidden_states(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Runs the backbone forward pass and retrieves sequence hidden states."""
        target_device = self.device
        input_ids = input_ids.to(target_device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(target_device)

        if hasattr(self, "embed"):
            # Standalone mode: pass padding mask to prevent attention over padded tokens
            x = self.embed(input_ids)
            has_padding = attention_mask is not None and bool((attention_mask == 0).any())
            padding_mask = (attention_mask == 0) if has_padding else None
            return self.backbone(x, src_key_padding_mask=padding_mask)
        else:
            # Hugging Face backbone
            outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
            if hasattr(outputs, "last_hidden_state"):
                return outputs.last_hidden_state
            return outputs[0]

    def evaluate_batch(
        self,
        batch: list[PackedSequence],
        pad_token_id: int = 0,
    ) -> list[dict[str, Result]]:
        """Evaluates a batch of packed sequences simultaneously in a single batched forward pass.

        Args:
            batch: List of PackedSequence objects.
            pad_token_id: Token ID used for padding sequences to uniform length.

        Returns:
            List of answer dictionaries, one per input packed sequence.
        """
        if not batch:
            return []

        self.eval()
        batch_size = len(batch)
        max_len = max(p.input_ids.shape[1] for p in batch)

        # Allocate padded tensors
        batched_input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long, device=self.device)
        batched_attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long, device=self.device)

        for b, p in enumerate(batch):
            seq_len = p.input_ids.shape[1]
            batched_input_ids[b, :seq_len] = p.input_ids[0].to(self.device)
            batched_attention_mask[b, :seq_len] = p.attention_mask[0].to(self.device)

        with torch.no_grad():
            hidden_states = self.extract_hidden_states(
                batched_input_ids, batched_attention_mask
            )  # [batch_size, max_len, hidden_size]

            head_dtype = next(self.noul_head.parameters()).dtype
            batch_answers: list[dict[str, Result]] = []

            for b, p in enumerate(batch):
                seq_hidden = hidden_states[b]
                if seq_hidden.dtype != head_dtype:
                    seq_hidden = seq_hidden.to(head_dtype)

                answers: dict[str, Result] = {}

                for q_id, span in p.questions.items():
                    if span.question_type == "noul":
                        h_query = seq_hidden[span.query_token_idx]
                        prob = self.noul_head(h_query).item()
                        answers[q_id] = NoulResult(noul=round(float(prob), 4))

                    elif span.question_type == "choice":
                        h_query = seq_hidden[span.query_token_idx]
                        h_opts = torch.stack([seq_hidden[idx] for idx in span.option_token_indices])
                        probs = self.choice_head(h_query, h_opts)
                        probs_list = probs.tolist()

                        prob_dict = {key: round(float(val), 4) for key, val in zip(span.option_keys, probs_list)}
                        best_idx = int(torch.argmax(probs).item())
                        best_key = span.option_keys[best_idx]

                        # Normalized Confidence: (K * P_max - 1) / (K - 1)
                        num_options = len(probs_list)
                        if num_options > 1:
                            p_max = float(torch.max(probs).item())
                            conf = (num_options * p_max - 1.0) / (num_options - 1.0)
                        else:
                            conf = 1.0

                        conf = 0.0 if math.isnan(conf) else max(0.0, min(1.0, conf))

                        answers[q_id] = ChoiceResult(
                            choice=best_key,
                            probabilities=prob_dict,
                            confidence=round(conf, 4),
                        )

                    elif span.question_type == "score":
                        h_query = seq_hidden[span.query_token_idx]
                        h_levels = torch.stack([seq_hidden[idx] for idx in span.option_token_indices])
                        expected_score, probs = self.score_head(h_query, h_levels)
                        probs_list = probs.tolist()

                        prob_dict = {key: round(float(val), 4) for key, val in zip(span.option_keys, probs_list)}

                        # Normalized Confidence across score levels: (K * P_max - 1) / (K - 1)
                        num_levels = len(probs_list)
                        if num_levels > 1:
                            p_max = float(torch.max(probs).item())
                            conf = (num_levels * p_max - 1.0) / (num_levels - 1.0)
                        else:
                            conf = 1.0

                        conf = 0.0 if math.isnan(conf) else max(0.0, min(1.0, conf))

                        answers[q_id] = ScoreResult(
                            score=round(float(expected_score.item()), 4),
                            probabilities=prob_dict,
                            confidence=round(conf, 4),
                        )

                batch_answers.append(answers)

            return batch_answers

    def evaluate_packed(self, packed: PackedSequence) -> dict[str, Result]:
        """Evaluates all questions in a packed sequence in a single forward pass."""
        return self.evaluate_batch([packed])[0]

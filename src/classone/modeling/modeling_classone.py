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
from classone.tokenizer import PackedSequence, are_questions_independent


def _round_probability_distribution(
    keys: list[str],
    raw_probs: list[float],
    decimals: int = 4,
) -> dict[str, float]:
    """Normalize and round a probability vector to fixed decimals while summing to 1.0.

    Uses largest-remainder allocation in scaled integer units so the rounded
    probabilities stay in [0.0, 1.0], preserve non-strict probability ordering
    (argmax consistency), and avoid cumulative rounding drift across up to 255 options.
    """
    n = len(keys)
    if n == 0:
        return {}

    cleaned = [float(p) if math.isfinite(p) and float(p) > 0.0 else 0.0 for p in raw_probs]
    total = sum(cleaned)
    if total <= 0.0:
        normalized = [1.0 / n] * n
    else:
        normalized = [p / total for p in cleaned]

    scale = 10**decimals
    scaled = [p * scale for p in normalized]
    ticks = [math.floor(val) for val in scaled]
    deficit = scale - sum(ticks)

    if deficit > 0:
        # Allocate remaining ticks by largest remainder; break ties by higher probability, then index.
        order = sorted(
            range(n),
            key=lambda i: (-(scaled[i] - ticks[i]), -normalized[i], i),
        )
        for i in order[:deficit]:
            ticks[i] += 1

    return {key: tick / scale for key, tick in zip(keys, ticks)}


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

    def forward(
        self, h_query: torch.Tensor, return_logits: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Args:

        h_query: [batch_size, hidden_size] or [hidden_size]
        return_logits: If True, returns (prob, logit)
        Returns:
            probability tensor in [0, 1] or (probability, unscaled logit)
        """
        if h_query.dim() == 1:
            h_query = h_query.unsqueeze(0)
        logit = self.net(h_query).squeeze(-1)
        prob = torch.sigmoid(logit / self.temperature)
        if return_logits:
            return prob, logit
        return prob


class ChoiceHead(nn.Module):
    """Categorical decision head computing calibrated distribution over dynamic options.

    Combines directional dot-product similarity with a bilinear interaction scorer
    [q; k; |q - k|; q * k] for expressive feature-level matching.
    """

    def __init__(self, hidden_size: int, head_hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.head_hidden_size = head_hidden_size
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

        # Bilinear interaction scorer over [q, k, |q - k|, q * k]
        self.scorer = nn.Sequential(
            nn.Linear(4 * head_hidden_size, head_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_size, 1),
        )
        self.temperature_raw = nn.Parameter(torch.zeros(1))

    @property
    def temperature(self) -> torch.Tensor:
        return F.softplus(self.temperature_raw) + 0.1

    def forward(
        self, h_query: torch.Tensor, h_options: torch.Tensor, return_logits: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Args:

        h_query: Query token vector [hidden_size] or [1, hidden_size]
        h_options: Candidate option token vectors [num_options, hidden_size]
        return_logits: If True, returns (probs, logits)
        Returns:
            Probability distribution over options [num_options] or (probs, unscaled logits)
        """
        if h_query.dim() == 1:
            h_query = h_query.unsqueeze(0)  # [1, hidden_size]

        q = self.q_proj(h_query)  # [1, head_hidden_size]
        k = self.k_proj(h_options)  # [num_options, head_hidden_size]

        num_options = k.shape[0]
        q_exp = q.expand(num_options, -1)

        # 1. Scaled dot-product similarity
        dot = torch.sum(q_exp * k, dim=-1) * self.scale

        # 2. Bilinear interaction features
        diff = torch.abs(q_exp - k)
        prod = q_exp * k
        features = torch.cat([q_exp, k, diff, prod], dim=-1)
        score_mlp = self.scorer(features).squeeze(-1)

        logits = dot + score_mlp
        probs = F.softmax(logits / self.temperature, dim=-1)
        if return_logits:
            return probs, logits
        return probs

    def load_state_dict(self, state_dict: dict[str, Any], strict: bool = True, assign: bool = False):
        """Loads state dict with backward compatibility for legacy checkpoints lacking bilinear scorer."""
        has_scorer = any(k.startswith("scorer.") for k in state_dict)
        if not has_scorer:
            strict = False
        return super().load_state_dict(state_dict, strict=strict, assign=assign)


class ScoreHead(nn.Module):
    """Ordinal decision head predicting expected score over an ordered rubric."""

    def __init__(self, hidden_size: int, head_hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.choice_evaluator = ChoiceHead(hidden_size, head_hidden_size, dropout)

    @property
    def temperature(self) -> torch.Tensor:
        return self.choice_evaluator.temperature

    def forward(
        self, h_query: torch.Tensor, h_levels: torch.Tensor, return_logits: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Args:

        h_query: Query token vector [hidden_size]
        h_levels: Rubric level vectors [num_levels, hidden_size]
        return_logits: If True, returns (expected_score, probabilities, logits)
        Returns:
            (expected_score, probabilities) or (expected_score, probabilities, logits)
        """
        if return_logits:
            probs, logits = self.choice_evaluator(h_query, h_levels, return_logits=True)
        else:
            probs = self.choice_evaluator(h_query, h_levels)
        num_levels = probs.shape[-1]
        # Match level_weights dtype to probs dtype to prevent Half/Float runtime crash
        level_weights = torch.arange(1, num_levels + 1, dtype=probs.dtype, device=probs.device)
        expected_score = torch.sum(probs * level_weights)
        if return_logits:
            return expected_score, probs, logits
        return expected_score, probs

    def load_state_dict(self, state_dict: dict[str, Any], strict: bool = True, assign: bool = False):
        """Loads state dict with backward compatibility for legacy checkpoints lacking bilinear scorer."""
        has_scorer = any(k.startswith("choice_evaluator.scorer.") for k in state_dict)
        if not has_scorer:
            strict = False
        return super().load_state_dict(state_dict, strict=strict, assign=assign)


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
        quantization: str | None = None,
        **backbone_kwargs,
    ) -> ClassOneModel:
        """Instantiates ClassOne with an existing HuggingFace backbone (e.g. Gemma 4 E2B).

        Args:
            base_model_name_or_path: HuggingFace model identifier, local directory, or existing nn.Module.
            config: Optional ClassOneConfig override.
            tokenizer: Optional tokenizer; if provided, backbone token embeddings are resized.
            torch_dtype: Precision dtype (e.g. torch.bfloat16, torch.float16, torch.float32).
            device: Target device ('mps', 'cuda', 'cpu'); if None, detects automatically.
            quantization: Quantization format ('8bit', '4bit', 'nf4'); loads backbone via bitsandbytes.
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

            cpu_quantize_8bit = False
            if quantization:
                from transformers import BitsAndBytesConfig

                quant_str = str(quantization).lower().replace("-", "")
                if quant_str in ("8bit", "int8", "q8"):
                    is_cpu_target = device == "cpu" or (device is None and not torch.cuda.is_available())
                    if is_cpu_target:
                        cpu_quantize_8bit = True
                        if "torch_dtype" not in kwargs:
                            kwargs["torch_dtype"] = torch.bfloat16
                    else:
                        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
                elif quant_str in ("4bit", "nf4", "int4"):
                    compute_dtype = torch_dtype or torch.float16
                    kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_compute_dtype=compute_dtype,
                    )
                else:
                    raise ValueError(f"Unsupported quantization: {quantization}. Choose '8bit' or '4bit'.")

                if "device_map" not in kwargs and device is not None and not cpu_quantize_8bit:
                    kwargs["device_map"] = str(device)
                elif "device_map" not in kwargs and not cpu_quantize_8bit:
                    kwargs["device_map"] = "auto"

            backbone = AutoModel.from_pretrained(model_name, **kwargs)

            if cpu_quantize_8bit:
                if hasattr(backbone, "vision_tower"):
                    backbone.vision_tower = None
                if hasattr(backbone, "audio_tower"):
                    backbone.audio_tower = None
                if hasattr(backbone, "get_input_embeddings"):
                    backbone.get_input_embeddings().float()
                if hasattr(backbone, "language_model") and hasattr(
                    backbone.language_model, "per_layer_model_projection"
                ):
                    backbone.language_model.per_layer_model_projection.float()

                layers_container = None
                if hasattr(backbone, "language_model") and hasattr(backbone.language_model, "layers"):
                    layers_container = backbone.language_model.layers
                elif hasattr(backbone, "layers"):
                    layers_container = backbone.layers

                if layers_container is not None:
                    for i, layer in enumerate(layers_container):
                        layer.float()
                        layers_container[i] = torch.ao.quantization.quantize_dynamic(
                            layer, {torch.nn.Linear}, dtype=torch.qint8
                        )
                else:
                    backbone = torch.ao.quantization.quantize_dynamic(backbone, {torch.nn.Linear}, dtype=torch.qint8)

        # Resize token embeddings only when tokenizer exceeds existing embedding rows
        if tokenizer is not None and hasattr(backbone, "resize_token_embeddings"):
            cur_embed = backbone.get_input_embeddings() if hasattr(backbone, "get_input_embeddings") else None
            cur_vocab = cur_embed.weight.shape[0] if cur_embed is not None and hasattr(cur_embed, "weight") else 0
            if cur_vocab == 0 or len(tokenizer) > cur_vocab:
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
        max_position_embeddings = getattr(text_config, "max_position_embeddings", None) or getattr(
            backbone.config, "max_position_embeddings", None
        )
        if isinstance(max_position_embeddings, bool) or not isinstance(max_position_embeddings, int):
            max_position_embeddings = None
        if max_position_embeddings is not None and max_position_embeddings <= 0:
            max_position_embeddings = None
        backbone_cls_cfg = getattr(backbone.config, "classone_config", None) or {}
        if isinstance(backbone_cls_cfg, dict) and backbone_cls_cfg:
            cls_eval_mode = str(backbone_cls_cfg.get("eval_mode", "stage1"))
            cls_m2_k = int(backbone_cls_cfg.get("m2_k_threshold", 64))
            cls_m2_top_k = int(backbone_cls_cfg.get("m2_top_k", 16))
        else:
            cls_eval_mode = "stage1"
            cls_m2_k = 64
            cls_m2_top_k = 16

        if config is None:
            config = ClassOneConfig(
                base_model_name_or_path=model_name,
                hidden_size=hidden_size,
                max_position_embeddings=max_position_embeddings,
                eval_mode=cls_eval_mode,
                m2_k_threshold=cls_m2_k,
                m2_top_k=cls_m2_top_k,
                classone_config=dict(backbone_cls_cfg) if backbone_cls_cfg else None,
            )
        else:
            config.hidden_size = hidden_size
            if getattr(config, "max_position_embeddings", None) is None and max_position_embeddings is not None:
                config.max_position_embeddings = max_position_embeddings
            if backbone_cls_cfg:
                if getattr(config, "classone_config", None) is None:
                    config.classone_config = dict(backbone_cls_cfg)
                if getattr(config, "eval_mode", "stage1") == "stage1" and "eval_mode" in backbone_cls_cfg:
                    config.eval_mode = cls_eval_mode
                if "m2_k_threshold" in backbone_cls_cfg:
                    config.m2_k_threshold = cls_m2_k
                if "m2_top_k" in backbone_cls_cfg:
                    config.m2_top_k = cls_m2_top_k

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

        is_quantized = getattr(backbone, "is_loaded_in_8bit", False) or getattr(backbone, "is_loaded_in_4bit", False)
        effective_dtype = torch_dtype
        if effective_dtype is None and not is_quantized and target_device.type != "cpu" and cls_eval_mode == "stage2":
            effective_dtype = torch.bfloat16

        if is_quantized:
            head_dtype = torch_dtype or torch.float16
            model.noul_head.to(device=target_device, dtype=head_dtype)
            model.choice_head.to(device=target_device, dtype=head_dtype)
            model.score_head.to(device=target_device, dtype=head_dtype)
        elif effective_dtype is not None:
            model.to(device=target_device, dtype=effective_dtype)
        else:
            model.to(target_device)

        return model

    def _backbone_float_dtype(self) -> torch.dtype:
        """Returns the floating-point dtype of the backbone for attention mask construction."""
        for p in self.backbone.parameters():
            if p.is_floating_point():
                return p.dtype
        return next(self.noul_head.parameters()).dtype

    def _wrap_4d_attention_mask(self, attention_mask: torch.Tensor | None) -> Any:
        """Wraps 4D attention masks for hybrid full/linear attention backbones (e.g. Qwen 3.5)."""
        if attention_mask is not None and attention_mask.ndim == 4:
            model_type = getattr(getattr(self.backbone, "config", None), "model_type", "").lower()
            if "qwen3_5" in model_type:
                return {"full_attention": attention_mask, "linear_attention": None}
        return attention_mask

    def build_position_invariant_mask(
        self,
        packed: PackedSequence,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Builds a 4D block-diagonal attention mask and aligned position IDs.

        Prevents candidate options from causally attending to earlier options in the prompt,
        eliminating option-order recency bias.
        """
        seq_len = packed.input_ids.shape[1]
        causal_2d = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=device))
        pos_ids = torch.arange(seq_len, dtype=torch.long, device=device)

        for q_id, span in packed.questions.items():
            spans = getattr(span, "option_token_spans", [])
            if len(spans) > 1:
                base_pos = spans[0][0]
                # Option isolation: mask cross-option attention
                for j in range(len(spans)):
                    s_j, e_j = spans[j]
                    # Symmetrize position IDs across options
                    pos_ids[s_j:e_j] = base_pos + torch.arange(e_j - s_j, dtype=torch.long, device=device)
                    for i in range(len(spans)):
                        if i != j:
                            s_i, e_i = spans[i]
                            causal_2d[s_j:e_j, s_i:e_i] = False

        mask_4d = torch.where(causal_2d, 0.0, float("-1e9")).to(dtype=dtype)[None, None, :, :]
        return mask_4d, pos_ids.unsqueeze(0)

    def extract_hidden_states(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Runs the backbone forward pass and retrieves sequence hidden states."""
        target_device = self.device
        input_ids = input_ids.to(target_device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(target_device)
        if position_ids is not None:
            position_ids = position_ids.to(target_device)

        if hasattr(self, "embed"):
            # Standalone mode: pass padding mask to prevent attention over padded tokens
            x = self.embed(input_ids)
            has_padding = attention_mask is not None and bool((attention_mask == 0).any())
            padding_mask = (attention_mask == 0) if has_padding else None
            return self.backbone(x, src_key_padding_mask=padding_mask)
        else:
            # Hugging Face backbone
            kwargs = {}
            if position_ids is not None:
                kwargs["position_ids"] = position_ids
            mask_arg = self._wrap_4d_attention_mask(attention_mask)
            outputs = self.backbone(input_ids=input_ids, attention_mask=mask_arg, **kwargs)
            if hasattr(outputs, "last_hidden_state"):
                hidden = outputs.last_hidden_state
            else:
                hidden = outputs[0]

            head_dtype = next(self.noul_head.parameters()).dtype
            if hidden.dtype != head_dtype:
                hidden = hidden.to(dtype=head_dtype)
            return hidden

    def _score_question_span(self, seq_hidden: torch.Tensor, span: Any) -> Result:
        """Evaluates a single question span against sequence hidden states."""
        if span.question_type == "noul":
            h_query = seq_hidden[span.query_token_idx]
            prob = self.noul_head(h_query).item()
            return NoulResult(noul=round(float(prob), 4))

        if span.question_type == "choice":
            h_query = seq_hidden[span.query_token_idx]
            h_opts = torch.stack([seq_hidden[idx] for idx in span.option_token_indices])
            probs = self.choice_head(h_query, h_opts)
            probs_list = probs.tolist()

            prob_dict = _round_probability_distribution(span.option_keys, probs_list, decimals=4)
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

            # Margin: delta between top-1 and top-2 candidate probabilities
            sorted_probs = sorted(probs_list, reverse=True)
            margin = float(sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) > 1 else 1.0
            is_decisive = conf >= 0.15

            return ChoiceResult(
                choice=best_key,
                probabilities=prob_dict,
                confidence=round(conf, 4),
                is_decisive=is_decisive,
                margin=round(margin, 4),
            )

        if span.question_type == "score":
            h_query = seq_hidden[span.query_token_idx]
            h_levels = torch.stack([seq_hidden[idx] for idx in span.option_token_indices])
            expected_score, probs = self.score_head(h_query, h_levels)
            probs_list = probs.tolist()

            prob_dict = _round_probability_distribution(span.option_keys, probs_list, decimals=4)

            # Normalized Confidence across score levels: (K * P_max - 1) / (K - 1)
            num_levels = len(probs_list)
            if num_levels > 1:
                p_max = float(torch.max(probs).item())
                conf = (num_levels * p_max - 1.0) / (num_levels - 1.0)
            else:
                conf = 1.0

            conf = 0.0 if math.isnan(conf) else max(0.0, min(1.0, conf))

            # Margin: delta between top-1 and top-2 level probabilities
            sorted_probs = sorted(probs_list, reverse=True)
            margin = float(sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) > 1 else 1.0
            is_decisive = conf >= 0.15

            return ScoreResult(
                score=round(float(expected_score.item()), 4),
                probabilities=prob_dict,
                confidence=round(conf, 4),
                is_decisive=is_decisive,
                margin=round(margin, 4),
            )

        raise ValueError(f"Unsupported question_type: {span.question_type}")

    def evaluate_batch(
        self,
        batch: list[PackedSequence],
        pad_token_id: int = 0,
        position_invariant: bool = True,
        max_2d_mask_tokens: int = 4096,
    ) -> list[dict[str, Result]]:
        """Evaluates a batch of packed sequences simultaneously in a single batched forward pass.

        Args:
            batch: List of PackedSequence objects.
            pad_token_id: Token ID used for padding sequences to uniform length.
            position_invariant: Whether to apply 2D block-diagonal option isolation mask and
                symmetrized option position IDs (default True).
            max_2d_mask_tokens: Maximum sequence length for allocating 4D attention masks on GPU.

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
            if len(batch) == 1 and position_invariant and not hasattr(self, "embed") and max_len <= max_2d_mask_tokens:
                has_options = any(
                    len(getattr(span, "option_token_spans", [])) > 1 for span in batch[0].questions.values()
                )
                if has_options:
                    mask_dtype = self._backbone_float_dtype()
                    mask_4d, pos_ids = self.build_position_invariant_mask(batch[0], self.device, dtype=mask_dtype)
                    hidden_states = self.extract_hidden_states(
                        batched_input_ids, attention_mask=mask_4d, position_ids=pos_ids
                    )
                else:
                    hidden_states = self.extract_hidden_states(batched_input_ids, batched_attention_mask)
            else:
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
                    answers[q_id] = self._score_question_span(seq_hidden, span)

                batch_answers.append(answers)

            return batch_answers

    def evaluate_packed(self, packed: PackedSequence, position_invariant: bool = True) -> dict[str, Result]:
        """Evaluates all questions in a packed sequence in a single forward pass."""
        return self.evaluate_batch([packed], position_invariant=position_invariant)[0]

    def evaluate_questions_branched(
        self,
        builder: Any,
        state: Any,
        questions: Any,
        options_in_prefix: bool | str = False,
        position_invariant: bool = True,
        max_2d_mask_tokens: int = 4096,
        max_tokens: int | None = None,
    ) -> tuple[dict[str, Result], int]:
        """Evaluates each question as a separate pass branched from the KV-cached shared state prefix."""
        import copy

        if len(questions) <= 1 or hasattr(self, "embed"):
            answers: dict[str, Result] = {}
            total_tokens = 0
            for q_id, q in questions.items():
                packed = builder.pack(
                    state=state,
                    questions={q_id: q},
                    options_in_prefix=options_in_prefix,
                )
                q_tokens = int(packed.input_ids.shape[1])
                if max_tokens is not None and q_tokens > max_tokens:
                    raise ValueError(
                        f"Prompt length ({q_tokens} tokens) exceeds maximum context length ({max_tokens})."
                    )
                total_tokens += q_tokens
                ans = self.evaluate_packed(packed, position_invariant=position_invariant)
                answers.update(ans)
            return answers, total_tokens

        self.eval()
        prefix_packed = builder.pack(
            state=state,
            questions={},
            options_in_prefix=options_in_prefix,
            preamble_questions=questions,
        )
        prefix_ids = prefix_packed.input_ids.to(self.device)
        prefix_len = int(prefix_ids.shape[1])
        if max_tokens is not None and prefix_len > max_tokens:
            raise ValueError(f"Prompt length ({prefix_len} tokens) exceeds maximum context length ({max_tokens}).")

        head_dtype = next(self.noul_head.parameters()).dtype
        mask_dtype = self._backbone_float_dtype()

        answers = {}
        total_tokens = prefix_len

        with torch.no_grad():
            base_out = self.backbone(
                input_ids=prefix_ids,
                attention_mask=torch.ones_like(prefix_ids),
                use_cache=True,
            )
            base_pkv = getattr(base_out, "past_key_values", None)
            if base_pkv is None:
                # Fallback if backbone does not return past_key_values
                total_tokens = 0
                for q_id, q in questions.items():
                    q_packed = builder.pack(
                        state=state,
                        questions={q_id: q},
                        options_in_prefix=options_in_prefix,
                        preamble_questions=questions,
                    )
                    q_tokens = int(q_packed.input_ids.shape[1])
                    if max_tokens is not None and q_tokens > max_tokens:
                        raise ValueError(
                            f"Prompt length ({q_tokens} tokens) exceeds maximum context length ({max_tokens})."
                        )
                    total_tokens += q_tokens
                    answers.update(self.evaluate_packed(q_packed, position_invariant=position_invariant))
                return answers, total_tokens

            for q_id, q in questions.items():
                q_packed = builder.pack(
                    state=state,
                    questions={q_id: q},
                    options_in_prefix=options_in_prefix,
                    preamble_questions=questions,
                )
                full_ids = q_packed.input_ids.to(self.device)
                total_len = int(full_ids.shape[1])
                if max_tokens is not None and total_len > max_tokens:
                    raise ValueError(
                        f"Prompt length ({total_len} tokens) exceeds maximum context length ({max_tokens})."
                    )

                suffix_ids = full_ids[:, prefix_len:]
                suffix_len = int(suffix_ids.shape[1])
                total_tokens += suffix_len

                pkv = copy.deepcopy(base_pkv)
                span = q_packed.questions[q_id]
                has_options = len(getattr(span, "option_token_spans", [])) > 1

                if position_invariant and has_options and total_len <= max_2d_mask_tokens:
                    mask_4d, pos_ids = self.build_position_invariant_mask(q_packed, self.device, dtype=mask_dtype)
                    suffix_mask_4d = mask_4d[:, :, prefix_len:, :]
                    suffix_pos_ids = pos_ids[:, prefix_len:]
                    mask_arg = self._wrap_4d_attention_mask(suffix_mask_4d)
                    q_out = self.backbone(
                        input_ids=suffix_ids,
                        attention_mask=mask_arg,
                        position_ids=suffix_pos_ids,
                        past_key_values=pkv,
                        use_cache=False,
                    )
                else:
                    full_mask = torch.ones((1, total_len), dtype=torch.long, device=self.device)
                    q_out = self.backbone(
                        input_ids=suffix_ids,
                        attention_mask=full_mask,
                        past_key_values=pkv,
                        use_cache=False,
                    )

                seq_hidden = q_out.last_hidden_state[0] if hasattr(q_out, "last_hidden_state") else q_out[0][0]
                if seq_hidden.dtype != head_dtype:
                    seq_hidden = seq_hidden.to(head_dtype)

                # Shift span indices by prefix_len to index into suffix seq_hidden
                shifted_span = copy.copy(span)
                shifted_span.query_token_idx = span.query_token_idx - prefix_len
                shifted_span.option_token_indices = [idx - prefix_len for idx in span.option_token_indices]
                answers[q_id] = self._score_question_span(seq_hidden, shifted_span)

        return answers, total_tokens

    def evaluate_request(
        self,
        builder: Any,
        state: Any,
        questions: Any,
        options_in_prefix: bool | str = False,
        position_invariant: bool = True,
        adaptive_branching: bool = True,
        max_tokens: int | None = None,
        eval_mode: str | None = None,
        m2_k_threshold: int | None = None,
        m2_top_k: int | None = None,
    ) -> tuple[dict[str, Result], int]:
        """Evaluates a full DecisionRequest using configured Stage 2 or Stage 1 defaults.

        - When `eval_mode == "stage2"` (or `self.config.eval_mode == "stage2"` /
          `ChatDecisionPromptBuilder`), routes to `evaluate_stage2_request` with LM-head
          letter readout and two-pass M2 shortlist routing (`m2_k_threshold`, `m2_top_k`).
        - Otherwise evaluates Stage 1 defaults (pos_inv_only_no_prefix + adaptive branching).
        """
        resolved_mode = (eval_mode or getattr(self.config, "eval_mode", "stage1")).strip().lower()
        if resolved_mode == "stage2" or hasattr(builder, "pack_question"):
            from classone.stage2_readout import ChatDecisionPromptBuilder, evaluate_stage2_request

            chat_builder = builder if hasattr(builder, "pack_question") else None
            if chat_builder is None and hasattr(builder, "tokenizer"):
                chat_builder = ChatDecisionPromptBuilder(builder.tokenizer)
            if chat_builder is None:
                raise ValueError("Stage 2 evaluation requires a ChatDecisionPromptBuilder or tokenizer-backed builder.")
            resolved_m2_k = int(
                m2_k_threshold if m2_k_threshold is not None else getattr(self.config, "m2_k_threshold", 64)
            )
            resolved_m2_top_k = int(m2_top_k if m2_top_k is not None else getattr(self.config, "m2_top_k", 16))
            return evaluate_stage2_request(
                self,
                chat_builder,
                state=state,
                questions=questions,
                m2_k_threshold=resolved_m2_k,
                m2_top_k=resolved_m2_top_k,
                max_tokens=max_tokens or 4096,
            )

        if adaptive_branching and are_questions_independent(state, questions):
            return self.evaluate_questions_branched(
                builder=builder,
                state=state,
                questions=questions,
                options_in_prefix=options_in_prefix,
                position_invariant=position_invariant,
                max_tokens=max_tokens,
            )

        packed = builder.pack(
            state=state,
            questions=questions,
            options_in_prefix=options_in_prefix,
        )
        input_token_count = int(packed.input_ids.shape[1])
        if max_tokens is not None and input_token_count > max_tokens:
            raise ValueError(
                f"Prompt length ({input_token_count} tokens) exceeds maximum context length ({max_tokens})."
            )
        answers = self.evaluate_packed(packed, position_invariant=position_invariant)
        return answers, input_token_count

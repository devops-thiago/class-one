"""Training and calibration loop for Jev System One decision models.

Implements:
- Parameter-Efficient Fine-Tuning (LoRA via PEFT)
- Multi-task RLCD training step with proper scoring rules (Brier + Log-Loss)
- Memory-efficient per-item gradient accumulation
- Temperature scaling calibration for optimal epistemic confidence
- Model checkpoint saving and loading (including LoRA adapters)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch
from peft import LoraConfig, PeftModel, get_peft_model

from classone.modeling.loss import RLCDLoss
from classone.modeling.modeling_classone import ClassOneModel
from classone.schemas import Question
from classone.tokenizer import ClassOnePromptBuilder


@dataclass
class TrainingItem:
    """A single training instance pairing input state, questions, and ground-truth targets."""

    state: str | dict[str, Any] | list[Any]
    questions: dict[str, Question]
    targets: dict[str, int | float | str]


class ClassOneTrainer:
    """Trainer executing Reinforcement Learning for Calibrated Decisions (RLCD)."""

    def __init__(
        self,
        model: ClassOneModel,
        prompt_builder: ClassOnePromptBuilder,
        loss_fn: RLCDLoss | None = None,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        device: str | torch.device | None = None,
    ):
        self.model = model
        self.prompt_builder = prompt_builder
        self.loss_fn = loss_fn or RLCDLoss(brier_weight=0.5, log_weight=1.0)
        self.lr = lr
        self.weight_decay = weight_decay
        self.device = torch.device(device) if device else self.model.device
        self.model.to(self.device)

        # Initialize optimizer for decision heads and trainable backbone parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(trainable_params, lr=self.lr, weight_decay=self.weight_decay)

    def enable_lora(
        self,
        r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        target_modules: list[str] | None = None,
    ):
        """Attaches LoRA adapters to the transformer backbone, keeping heads trainable."""
        if target_modules is None:
            # Common attention and MLP projection layers in Gemma / modern LLMs
            target_candidates = [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]
            if hasattr(self.model.backbone, "language_model"):
                # Multi-modal architectures like Gemma 4: target linear projections in language_model
                target_modules = r".*language_model.*(q_proj|o_proj|gate_proj|up_proj|down_proj).*"
            else:
                # Filter to candidates that actually exist as Linear in the backbone
                existing_modules = {
                    name.split(".")[-1]
                    for name, mod in self.model.backbone.named_modules()
                    if isinstance(mod, torch.nn.Linear)
                }
                matched = [m for m in target_candidates if m in existing_modules]
                target_modules = matched if matched else target_candidates

        lora_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none",
        )

        # Apply LoRA to the backbone
        self.model.backbone = get_peft_model(self.model.backbone, lora_config)

        # Ensure decision heads remain trainable
        for head in [
            self.model.noul_head,
            self.model.choice_head,
            self.model.score_head,
        ]:
            for param in head.parameters():
                param.requires_grad = True

        # Re-initialize optimizer with updated trainable parameters preserving initial lr & decay
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(trainable_params, lr=self.lr, weight_decay=self.weight_decay)

    def train_step(self, batch: list[TrainingItem]) -> dict[str, float]:
        """Executes a single gradient update across a batch of multi-task items with per-item accumulation."""
        self.model.train()
        self.optimizer.zero_grad()

        # Count total valid questions in batch first for correct gradient normalization
        total_valid_questions = 0
        packed_batch = []
        for item in batch:
            packed = self.prompt_builder.pack(state=item.state, questions=item.questions)
            packed_batch.append((item, packed))
            for q_id in item.targets:
                if q_id in packed.questions:
                    total_valid_questions += 1

        if total_valid_questions == 0:
            return {"loss": 0.0, "questions": 0}

        accumulated_loss_val = 0.0

        for item, packed in packed_batch:
            input_ids = packed.input_ids.to(self.device)
            attention_mask = packed.attention_mask.to(self.device)

            hidden_states = self.model.extract_hidden_states(input_ids=input_ids, attention_mask=attention_mask)
            seq_hidden = hidden_states[0]
            head_dtype = next(self.model.noul_head.parameters()).dtype
            if seq_hidden.dtype != head_dtype:
                seq_hidden = seq_hidden.to(head_dtype)

            item_loss = torch.tensor(0.0, device=self.device)
            item_question_count = 0

            for q_id, target in item.targets.items():
                if q_id not in packed.questions:
                    continue

                span = packed.questions[q_id]

                if span.question_type == "noul":
                    h_query = seq_hidden[span.query_token_idx]
                    prob = self.model.noul_head(h_query)
                    target_tensor = torch.tensor([float(target)], device=self.device, dtype=torch.float32)
                    q_loss = self.loss_fn.forward_binary(prob, target_tensor)
                    item_loss = item_loss + q_loss
                    item_question_count += 1

                elif span.question_type == "choice":
                    h_query = seq_hidden[span.query_token_idx]
                    h_opts = torch.stack([seq_hidden[idx] for idx in span.option_token_indices])
                    probs = self.model.choice_head(h_query, h_opts)  # [num_options]

                    # Target index resolution
                    if isinstance(target, str):
                        target_idx = span.option_keys.index(target) if target in span.option_keys else 0
                    else:
                        target_idx = int(target)

                    target_tensor = torch.tensor([target_idx], device=self.device, dtype=torch.long)
                    q_loss = self.loss_fn.forward_multiclass(probs.unsqueeze(0), target_tensor)
                    item_loss = item_loss + q_loss
                    item_question_count += 1

                elif span.question_type == "score":
                    h_query = seq_hidden[span.query_token_idx]
                    h_levels = torch.stack([seq_hidden[idx] for idx in span.option_token_indices])
                    _, probs = self.model.score_head(h_query, h_levels)

                    # Robust score target parsing (supports numeric index, string level number, or rubric text)
                    if isinstance(target, (int, float)):
                        target_idx = max(0, min(len(span.option_keys) - 1, int(target) - 1))
                    elif str(target) in span.option_keys:
                        target_idx = span.option_keys.index(str(target))
                    else:
                        target_idx = 0

                    target_tensor = torch.tensor([target_idx], device=self.device, dtype=torch.long)
                    q_loss = self.loss_fn.forward_multiclass(probs.unsqueeze(0), target_tensor)
                    item_loss = item_loss + q_loss
                    item_question_count += 1

            if item_question_count > 0:
                scaled_loss = item_loss / total_valid_questions
                scaled_loss.backward()
                accumulated_loss_val += float(item_loss.item())

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return {
            "loss": accumulated_loss_val / total_valid_questions,
            "questions": total_valid_questions,
        }

    def calibrate_temperature(
        self,
        val_data: list[TrainingItem],
        lr: float = 0.05,
        max_epochs: int = 20,
    ) -> dict[str, float]:
        """Calibrates head temperatures via validation loss optimization to minimize ECE."""
        self.model.eval()

        # Cache original parameter requires_grad states to prevent unfreezing the backbone
        original_grad_states = {name: p.requires_grad for name, p in self.model.named_parameters()}

        for p in self.model.parameters():
            p.requires_grad = False

        # Optimize the raw temperature parameters
        self.model.noul_head.temperature_raw.requires_grad = True
        self.model.choice_head.temperature_raw.requires_grad = True
        self.model.score_head.choice_evaluator.temperature_raw.requires_grad = True

        temp_optimizer = torch.optim.Adam(
            [
                self.model.noul_head.temperature_raw,
                self.model.choice_head.temperature_raw,
                self.model.score_head.choice_evaluator.temperature_raw,
            ],
            lr=lr,
        )

        for _ in range(max_epochs):
            temp_optimizer.zero_grad()
            calib_loss = torch.tensor(0.0, device=self.device)
            count = 0

            for item in val_data:
                packed = self.prompt_builder.pack(item.state, item.questions)
                with torch.no_grad():
                    hidden_states = self.model.extract_hidden_states(
                        packed.input_ids.to(self.device),
                        packed.attention_mask.to(self.device),
                    )
                    seq_hidden = hidden_states[0]

                for q_id, target in item.targets.items():
                    if q_id not in packed.questions:
                        continue
                    span = packed.questions[q_id]
                    if span.question_type == "noul":
                        h_query = seq_hidden[span.query_token_idx]
                        prob = self.model.noul_head(h_query)
                        t = torch.tensor([float(target)], device=self.device)
                        calib_loss = calib_loss + self.loss_fn.forward_binary(prob, t)
                        count += 1
                    elif span.question_type == "choice":
                        h_query = seq_hidden[span.query_token_idx]
                        h_opts = torch.stack([seq_hidden[idx] for idx in span.option_token_indices])
                        probs = self.model.choice_head(h_query, h_opts)
                        t_idx = (
                            span.option_keys.index(target)
                            if (isinstance(target, str) and target in span.option_keys)
                            else int(target)
                        )
                        t = torch.tensor([t_idx], device=self.device)
                        calib_loss = calib_loss + self.loss_fn.forward_multiclass(probs.unsqueeze(0), t)
                        count += 1
                    elif span.question_type == "score":
                        h_query = seq_hidden[span.query_token_idx]
                        h_levels = torch.stack([seq_hidden[idx] for idx in span.option_token_indices])
                        _, probs = self.model.score_head(h_query, h_levels)
                        if isinstance(target, (int, float)):
                            t_idx = max(0, min(len(span.option_keys) - 1, int(target) - 1))
                        elif str(target) in span.option_keys:
                            t_idx = span.option_keys.index(str(target))
                        else:
                            t_idx = 0
                        t = torch.tensor([t_idx], device=self.device)
                        calib_loss = calib_loss + self.loss_fn.forward_multiclass(probs.unsqueeze(0), t)
                        count += 1

            if count > 0:
                (calib_loss / count).backward()
                temp_optimizer.step()

        # Accurately restore original gradient states
        for name, p in self.model.named_parameters():
            p.requires_grad = original_grad_states[name]

        return {
            "noul_temp": float(self.model.noul_head.temperature.item()),
            "choice_temp": float(self.model.choice_head.temperature.item()),
            "score_temp": float(self.model.score_head.temperature.item()),
        }

    def save_checkpoint(self, output_dir: str):
        """Saves model weights, decision heads, and configuration to directory."""
        os.makedirs(output_dir, exist_ok=True)
        checkpoint_path = os.path.join(output_dir, "classone_heads.pt")
        torch.save(
            {
                "noul_head": self.model.noul_head.state_dict(),
                "choice_head": self.model.choice_head.state_dict(),
                "score_head": self.model.score_head.state_dict(),
                "config": self.model.config.to_dict(),
            },
            checkpoint_path,
        )

        # If LoRA was attached, save adapters
        if hasattr(self.model.backbone, "save_pretrained"):
            lora_dir = os.path.join(output_dir, "lora_backbone")
            self.model.backbone.save_pretrained(lora_dir)

    def load_checkpoint(self, checkpoint_dir: str):
        """Loads decision heads and optional LoRA adapters from directory."""
        checkpoint_path = os.path.join(checkpoint_dir, "classone_heads.pt")
        if not os.path.exists(checkpoint_path):
            checkpoint_path = os.path.join(checkpoint_dir, "jev_heads.pt")
        data = torch.load(checkpoint_path, map_location=self.device)
        self.model.noul_head.load_state_dict(data["noul_head"])
        self.model.choice_head.load_state_dict(data["choice_head"])
        self.model.score_head.load_state_dict(data["score_head"])

        lora_dir = os.path.join(checkpoint_dir, "lora_backbone")
        if os.path.exists(lora_dir):
            if isinstance(self.model.backbone, PeftModel):
                self.model.backbone.load_adapter(lora_dir, adapter_name="default")
            else:
                self.model.backbone = PeftModel.from_pretrained(self.model.backbone, lora_dir)

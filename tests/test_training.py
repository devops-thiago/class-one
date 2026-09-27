"""Unit and integration tests for ClassOne RLCD trainer, LoRA, and calibration."""

import os
import tempfile

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import GemmaConfig, GemmaModel, PreTrainedTokenizerFast

from classone.modeling.configuration_classone import ClassOneConfig
from classone.modeling.modeling_classone import ClassOneModel
from classone.schemas import ChoiceQuestion, NoulQuestion, ScoreQuestion
from classone.tokenizer import ClassOnePromptBuilder
from classone.train.trainer import ClassOneTrainer, TrainingItem


@pytest.fixture
def trainer_setup():
    vocab = {"[UNK]": 0, "[PAD]": 1, "[CLS]": 2, "[SEP]": 3, "[MASK]": 4}
    for i in range(50):
        vocab[f"tok_{i}"] = 5 + i

    tok = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    fast_tok = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]", pad_token="[PAD]")
    builder = ClassOnePromptBuilder(fast_tok)

    config = ClassOneConfig(hidden_size=64, head_hidden_size=32)
    model = ClassOneModel(config)

    trainer = ClassOneTrainer(
        model=model,
        prompt_builder=builder,
        lr=1e-3,
        device="cpu",
    )

    return trainer, builder, model


def test_train_step_gradient_update(trainer_setup):
    trainer, *_ = trainer_setup

    batch = [
        TrainingItem(
            state="I want a refund for transaction #1234",
            questions={
                "is_refund": NoulQuestion(instructions="Is this asking for a refund?"),
                "dept": ChoiceQuestion(
                    instructions="Routing queue",
                    criteria={"billing": "Payment & refund", "tech": "Bugs & code"},
                ),
                "urgency": ScoreQuestion(
                    instructions="Ticket urgency",
                    criteria=["low", "urgent"],
                ),
            },
            targets={
                "is_refund": 1,
                "dept": "billing",
                "urgency": 2,
            },
        ),
        TrainingItem(
            state="The mobile app crashed on startup.",
            questions={
                "is_refund": NoulQuestion(instructions="Is this asking for a refund?"),
                "dept": ChoiceQuestion(
                    instructions="Routing queue",
                    criteria={"billing": "Payment & refund", "tech": "Bugs & code"},
                ),
            },
            targets={
                "is_refund": 0,
                "dept": "tech",
            },
        ),
    ]

    metrics = trainer.train_step(batch)
    assert "loss" in metrics
    assert metrics["loss"] > 0.0
    assert metrics["questions"] == 5


def test_temperature_calibration(trainer_setup):
    trainer, *_ = trainer_setup

    val_data = [
        TrainingItem(
            state="Payment error code 500",
            questions={
                "is_error": NoulQuestion(instructions="Is this an error?"),
                "category": ChoiceQuestion(
                    instructions="Category",
                    criteria={"err": "Error", "info": "Info"},
                ),
            },
            targets={
                "is_error": 1,
                "category": "err",
            },
        )
    ]

    temps = trainer.calibrate_temperature(val_data, lr=0.1, max_epochs=5)
    assert "noul_temp" in temps
    assert "choice_temp" in temps
    assert temps["noul_temp"] > 0.0


def test_save_and_load_checkpoint(trainer_setup):
    trainer, _, model = trainer_setup

    with tempfile.TemporaryDirectory() as tmpdir:
        trainer.save_checkpoint(tmpdir)
        assert os.path.exists(os.path.join(tmpdir, "classone_heads.pt"))

        # Mutate head weight
        with torch.no_grad():
            model.noul_head.net[0].weight.fill_(99.0)

        # Load back
        trainer.load_checkpoint(tmpdir)
        # Verify weight is restored (not 99.0)
        assert not torch.allclose(
            model.noul_head.net[0].weight,
            torch.full_like(model.noul_head.net[0].weight, 99.0),
        )


def test_enable_lora_on_gemma():
    vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}
    tok = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    tok.pre_tokenizer = Whitespace()
    fast_tok = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="<unk>", pad_token="<pad>")
    builder = ClassOnePromptBuilder(fast_tok)

    gemma_cfg = GemmaConfig(
        vocab_size=len(fast_tok),
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=32,
    )
    gemma_backbone = GemmaModel(gemma_cfg)
    classone_gemma = ClassOneModel.from_backbone(
        base_model_name_or_path=gemma_backbone,
        tokenizer=fast_tok,
        device="cpu",
    )

    trainer = ClassOneTrainer(
        model=classone_gemma,
        prompt_builder=builder,
        device="cpu",
    )

    # Attach LoRA
    trainer.enable_lora(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"])

    # Verify LoRA parameters exist in backbone
    lora_params = [name for name, param in classone_gemma.backbone.named_parameters() if "lora" in name]
    assert len(lora_params) > 0

    # Ensure decision heads are still trainable
    for p in classone_gemma.noul_head.parameters():
        assert p.requires_grad is True

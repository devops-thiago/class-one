"""Unit tests for the Jev dataset pipeline, task converters, and DataLoader integration."""

import os
import tempfile

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from classone.data.dataset import (
    ClassOneDataset,
    bundle_multi_task_items,
    create_choice_item,
    create_classone_dataloader,
    create_noul_item,
    create_score_item,
)
from classone.modeling.configuration_classone import ClassOneConfig
from classone.modeling.modeling_classone import ClassOneModel
from classone.schemas import ChoiceQuestion, NoulQuestion, ScoreQuestion
from classone.tokenizer import ClassOnePromptBuilder
from classone.train.trainer import ClassOneTrainer


def test_create_noul_item():
    item = create_noul_item(
        state="Is this transaction suspicious?",
        instruction="Check for fraud indicators",
        target=True,
        question_id="fraud",
    )
    assert item.questions["fraud"].type == "noul"
    assert item.targets["fraud"] == 1.0


def test_create_choice_item():
    all_categories = {
        "refund": "Customer asks for refund",
        "tech": "Technical bug or crash",
        "sales": "Pricing and enterprise upgrades",
        "legal": "Terms of service and privacy",
    }
    # With distractor subsampling
    item = create_choice_item(
        state="I want my money back",
        instruction="Select department",
        target_key="refund",
        all_criteria=all_categories,
        max_options=3,
        question_id="dept",
    )
    choice_q = item.questions["dept"]
    assert choice_q.type == "choice"
    assert "refund" in choice_q.criteria
    assert len(choice_q.criteria) == 3
    assert item.targets["dept"] == "refund"


def test_create_score_item():
    rubric = ["calm", "annoyed", "furious"]
    item = create_score_item(
        state="Everything is broken!",
        instruction="Rate customer frustration",
        target_level=3,
        rubric=rubric,
        question_id="frustration",
    )
    score_q = item.questions["frustration"]
    assert score_q.type == "score"
    assert len(score_q.criteria) == 3
    assert item.targets["frustration"] == 3


def test_jsonl_serialization_and_dataloader():
    items = [
        create_noul_item("Text A", "Is spam?", False, "q1"),
        create_choice_item("Text B", "Category?", "cat_1", {"cat_1": "C1", "cat_2": "C2"}, question_id="q2"),
        create_score_item("Text C", "Score?", 2, ["low", "high"], question_id="q3"),
    ]
    dataset = ClassOneDataset(items)
    assert len(dataset) == 3

    with tempfile.TemporaryDirectory() as tmpdir:
        jsonl_path = os.path.join(tmpdir, "dataset.jsonl")
        dataset.save_jsonl(jsonl_path)
        assert os.path.exists(jsonl_path)

        loaded_dataset = ClassOneDataset.load_jsonl(jsonl_path)
        assert len(loaded_dataset) == 3
        assert loaded_dataset[0].questions["q1"].type == "noul"
        assert loaded_dataset[1].questions["q2"].type == "choice"
        assert loaded_dataset[2].questions["q3"].type == "score"

    loader = create_classone_dataloader(dataset, batch_size=2, shuffle=False)
    batches = list(loader)
    assert len(batches) == 2
    assert len(batches[0]) == 2
    assert len(batches[1]) == 1


def test_dataloader_to_trainer_end_to_end():
    vocab = {"[UNK]": 0, "[PAD]": 1, "[CLS]": 2, "[SEP]": 3, "[MASK]": 4}
    for i in range(20):
        vocab[f"w_{i}"] = 5 + i
    tok = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    fast_tok = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]", pad_token="[PAD]")
    builder = ClassOnePromptBuilder(fast_tok)
    config = ClassOneConfig(hidden_size=64, head_hidden_size=32)
    model = ClassOneModel(config)
    trainer = ClassOneTrainer(model=model, prompt_builder=builder, device="cpu")

    # Multi-task item
    multi_task_item = bundle_multi_task_items(
        state="Customer account was compromised.",
        questions={
            "urgent": NoulQuestion(instructions="Is this urgent?"),
            "action": ChoiceQuestion(
                instructions="Action required",
                criteria={"lock": "Lock account", "notify": "Send notification"},
            ),
            "severity": ScoreQuestion(
                instructions="Severity level",
                criteria=["low", "medium", "critical"],
            ),
        },
        targets={
            "urgent": 1,
            "action": "lock",
            "severity": 3,
        },
    )

    dataset = ClassOneDataset([multi_task_item, multi_task_item])
    loader = create_classone_dataloader(dataset, batch_size=2)

    for batch in loader:
        metrics = trainer.train_step(batch)
        assert metrics["loss"] > 0.0
        assert metrics["questions"] == 6  # 3 questions * 2 items in batch

"""Dataset pipeline and task formatters for Jev System One training.

Provides:
- Converters for Intent/Routing datasets into Choice questions
- Converters for Moderation/Verification datasets into Noul questions
- Converters for Rating/Severity datasets into Score questions
- Multi-task sample combiners (merging multiple questions per state)
- PyTorch Dataset and DataLoader integration
"""

from __future__ import annotations

import json
import random
from typing import Any

from torch.utils.data import DataLoader, Dataset

from classone.schemas import ChoiceQuestion, NoulQuestion, Question, ScoreQuestion
from classone.train.trainer import TrainingItem


def create_noul_item(
    state: str | dict[str, Any] | list[Any],
    instruction: str,
    target: bool | float,
    question_id: str = "q_noul",
) -> TrainingItem:
    """Creates a TrainingItem with a single Noul boolean verification question."""
    target_val = 1.0 if target is True or target == 1 else (0.0 if target is False or target == 0 else float(target))
    return TrainingItem(
        state=state,
        questions={question_id: NoulQuestion(instructions=instruction)},
        targets={question_id: target_val},
    )


def create_choice_item(
    state: str | dict[str, Any] | list[Any],
    instruction: str,
    target_key: str,
    all_criteria: dict[str, str],
    max_options: int | None = None,
    question_id: str = "q_choice",
) -> TrainingItem:
    """Creates a TrainingItem with a Choice classification question.

    If max_options is specified and smaller than total categories, subsamples
    distractor options while always including the target_key.
    """
    if target_key not in all_criteria:
        raise ValueError(f"target_key '{target_key}' not found in all_criteria keys.")

    if max_options and max_options < len(all_criteria):
        distractor_keys = [k for k in all_criteria if k != target_key]
        num_distractors = min(len(distractor_keys), max(1, max_options - 1))
        sampled_keys = random.sample(distractor_keys, num_distractors) + [target_key]
        random.shuffle(sampled_keys)
        criteria = {k: all_criteria[k] for k in sampled_keys}
    else:
        criteria = dict(all_criteria)

    return TrainingItem(
        state=state,
        questions={
            question_id: ChoiceQuestion(
                instructions=instruction,
                criteria=criteria,
            )
        },
        targets={question_id: target_key},
    )


def create_score_item(
    state: str | dict[str, Any] | list[Any],
    instruction: str,
    target_level: float | str,
    rubric: list[str],
    question_id: str = "q_score",
) -> TrainingItem:
    """Creates a TrainingItem with an ordinal Score rubric question."""
    return TrainingItem(
        state=state,
        questions={
            question_id: ScoreQuestion(
                instructions=instruction,
                criteria=rubric,
            )
        },
        targets={question_id: target_level},
    )


def bundle_multi_task_items(
    state: str | dict[str, Any] | list[Any],
    questions: dict[str, Question],
    targets: dict[str, int | float | str],
) -> TrainingItem:
    """Bundles multiple parallel questions over a single shared input state."""
    return TrainingItem(state=state, questions=questions, targets=targets)


class ClassOneDataset(Dataset):
    """PyTorch Dataset containing ClassOne TrainingItems."""

    def __init__(self, items: list[TrainingItem]):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> TrainingItem:
        return self.items[idx]

    def save_jsonl(self, path: str):
        """Exports dataset items to JSONL format."""
        with open(path, "w", encoding="utf-8") as f:
            for item in self.items:
                serialized_questions = {}
                for q_id, q in item.questions.items():
                    serialized_questions[q_id] = q.model_dump()

                row = {
                    "state": item.state,
                    "questions": serialized_questions,
                    "targets": item.targets,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    @classmethod
    def load_jsonl(cls, path: str) -> ClassOneDataset:
        """Loads dataset from JSONL format."""
        items: list[TrainingItem] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                questions: dict[str, Question] = {}
                for q_id, q_data in row["questions"].items():
                    q_type = q_data.get("type")
                    if q_type == "noul":
                        questions[q_id] = NoulQuestion(**q_data)
                    elif q_type == "choice":
                        questions[q_id] = ChoiceQuestion(**q_data)
                    elif q_type == "score":
                        questions[q_id] = ScoreQuestion(**q_data)
                items.append(
                    TrainingItem(
                        state=row["state"],
                        questions=questions,
                        targets=row["targets"],
                    )
                )
        return cls(items)


def classone_collate_fn(batch: list[TrainingItem]) -> list[TrainingItem]:
    """Collate function for PyTorch DataLoader preserving List[TrainingItem]."""
    return batch


def create_classone_dataloader(
    dataset: ClassOneDataset,
    batch_size: int = 4,
    shuffle: bool = True,
) -> DataLoader:
    """Creates a PyTorch DataLoader yielding batches of TrainingItem."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=classone_collate_fn,
    )

"""ClassOne: Open-source System 1 Decision Model Architecture built on Gemma 4 and Qwen 3.5."""

__version__ = "0.3.0"

from classone.modeling import ClassOneConfig, ClassOneModel
from classone.schemas import (
    ChoiceQuestion,
    ChoiceResult,
    DecisionRequest,
    DecisionResponse,
    DecisionUsage,
    NoulQuestion,
    NoulResult,
    ScoreQuestion,
    ScoreResult,
)
from classone.sdk import (
    AsyncClassOneClient,
    Choice,
    ChoiceAnswer,
    ClassOneClient,
    ClassOneResponse,
    Noul,
    NoulAnswer,
    Score,
    ScoreAnswer,
)
from classone.stage2_readout import (
    ChatDecisionPromptBuilder,
    ChatPackedQuestion,
    build_single_token_labels,
    compute_lm_letter_logits,
    evaluate_stage2_request,
    load_stage2_snapshot,
)

__all__ = [
    "ChatDecisionPromptBuilder",
    "ChatPackedQuestion",
    "Choice",
    "ChoiceAnswer",
    "ChoiceQuestion",
    "ChoiceResult",
    "ClassOneClient",
    "AsyncClassOneClient",
    "ClassOneConfig",
    "ClassOneModel",
    "ClassOneResponse",
    "DecisionRequest",
    "DecisionResponse",
    "DecisionUsage",
    "Noul",
    "NoulAnswer",
    "NoulQuestion",
    "NoulResult",
    "Score",
    "ScoreAnswer",
    "ScoreQuestion",
    "ScoreResult",
    "build_single_token_labels",
    "compute_lm_letter_logits",
    "evaluate_stage2_request",
    "load_stage2_snapshot",
]

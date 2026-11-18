"""Unit tests for Stage 1: c2 default layout (options-in-prefix + 2D mask) and adaptive branching."""

from __future__ import annotations

from types import SimpleNamespace

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from classone.modeling.configuration_classone import ClassOneConfig
from classone.modeling.modeling_classone import ClassOneModel
from classone.schemas import ChoiceQuestion, NoulQuestion, ScoreQuestion
from classone.tokenizer import ClassOnePromptBuilder, are_questions_independent, has_substantive_options


def _build_test_tokenizer() -> PreTrainedTokenizerFast:
    vocab = {
        "[UNK]": 0,
        "[PAD]": 1,
        "State": 2,
        "A": 3,
        "B": 4,
        "C": 5,
        "yes": 6,
        "no": 7,
        "Option": 8,
        "Contract": 9,
        "clause": 10,
    }
    tok = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]", pad_token="[PAD]")


def test_pos_inv_no_prefix_is_default_and_substantive_options_in_prefix_skips_boilerplate_booleans() -> None:
    tok = _build_test_tokenizer()
    builder = ClassOnePromptBuilder(tok)
    q_substantive = ChoiceQuestion(
        instructions="Select the matching intent.",
        criteria={"opt_a": "Activate card", "opt_b": "Cancel transfer", "opt_c": "Check balance"},
    )
    q_boolean = ChoiceQuestion(
        instructions="Should the assistant invoke this tool?",
        criteria={"yes": "Invoke this tool.", "no": "Do not invoke this tool."},
    )

    # 1. Default is options_in_prefix=False (pos_inv_only_no_prefix)
    packed_default = builder.pack(state="User wants to check balance.", questions={"q1": q_substantive})
    assert "Candidate Options:" not in packed_default.text
    assert len(packed_default.questions["q1"].option_token_spans) == 3

    # 2. Enabling options_in_prefix=True lists substantive options
    packed_substantive = builder.pack(
        state="User wants to check balance.",
        questions={"q1": q_substantive},
        options_in_prefix=True,
    )
    assert (
        "Candidate Options:\n- opt_a: Activate card\n- opt_b: Cancel transfer\n- opt_c: Check balance"
        in packed_substantive.text
    )
    assert packed_substantive.input_ids.shape[1] > packed_default.input_ids.shape[1]

    # 3. Boilerplate boolean yes/no options are NEVER repeated in Candidate Options even when options_in_prefix=True
    assert has_substantive_options([("yes", "Invoke this tool."), ("no", "Do not invoke this tool.")]) is False
    assert has_substantive_options([("true", "True"), ("false", "False")]) is False
    assert has_substantive_options([("A", "Activate card"), ("B", "Cancel transfer")]) is True
    assert has_substantive_options([(f"opt_{i}", f"Category {i}") for i in range(25)]) is False
    packed_bool = builder.pack(
        state="Tool call check.",
        questions={"q_bool": q_boolean},
        options_in_prefix=True,
    )
    assert "Candidate Options:" not in packed_bool.text


def test_2d_position_invariant_mask_isolates_options_and_symmetrizes_positions() -> None:
    tok = _build_test_tokenizer()
    builder = ClassOnePromptBuilder(tok)
    config = ClassOneConfig(hidden_size=64, head_hidden_size=32)
    model = ClassOneModel(config)

    q = ChoiceQuestion(
        instructions="Pick one.",
        criteria={"A": "First option description", "B": "Second option description", "C": "Third option description"},
    )
    packed = builder.pack(state="Shared context state.", questions={"q1": q})
    spans = packed.questions["q1"].option_token_spans
    query_idx = packed.questions["q1"].query_token_idx
    assert len(spans) == 3

    mask_4d, pos_ids = model.build_position_invariant_mask(packed, device=torch.device("cpu"), dtype=torch.float32)
    assert mask_4d.shape == (1, 1, packed.input_ids.shape[1], packed.input_ids.shape[1])
    assert pos_ids.shape == (1, packed.input_ids.shape[1])

    s0, e0 = spans[0]
    s1, e1 = spans[1]
    s2, e2 = spans[2]

    # Option 1 and Option 2 must NOT attend to Option 0
    assert torch.all(mask_4d[0, 0, s1:e1, s0:e0] < -1e8)
    assert torch.all(mask_4d[0, 0, s2:e2, s0:e0] < -1e8)
    assert torch.all(mask_4d[0, 0, s2:e2, s1:e1] < -1e8)

    # Each option attends causally to itself and to the prefix before s0
    assert float(mask_4d[0, 0, s1, s1].item()) == 0.0
    assert float(mask_4d[0, 0, s2, s0 - 1].item()) == 0.0

    # The query token (<|choice_end|>) must still attend to all option spans
    assert float(mask_4d[0, 0, query_idx, e0 - 1].item()) == 0.0
    assert float(mask_4d[0, 0, query_idx, e1 - 1].item()) == 0.0
    assert float(mask_4d[0, 0, query_idx, e2 - 1].item()) == 0.0

    # Position IDs at the start of each option span must be identical (equal to s0)
    assert int(pos_ids[0, s0].item()) == s0
    assert int(pos_ids[0, s1].item()) == s0
    assert int(pos_ids[0, s2].item()) == s0


def test_are_questions_independent_structural_rules() -> None:
    long_state = (
        "This Non-Disclosure Agreement is entered into by and between Disclosing Party and Receiving Party. "
        "All confidential materials shared under this Agreement shall remain protected for five (5) years "
        "from the date of disclosure, except where information is independently developed or publicly known."
    )

    # 1. Single question -> False (no branching needed)
    q_single = {
        "nda_1": ChoiceQuestion(
            instructions="Does the contract allow sharing with third parties?",
            criteria={"Entailment": "Yes", "Contradiction": "No", "NotMentioned": "Not mentioned"},
        )
    }
    assert are_questions_independent(long_state, q_single) is False

    # 2. Independent multi-class contract / entity questions over shared state -> True
    q_independent = {
        "nda_1": ChoiceQuestion(
            instructions="Does the contract protect confidential materials for five years?",
            criteria={"Entailment": "Entailed", "Contradiction": "Contradicted", "NotMentioned": "Not mentioned"},
        ),
        "nda_2": ChoiceQuestion(
            instructions="Are independently developed materials excluded from confidentiality?",
            criteria={"Entailment": "Entailed", "Contradiction": "Contradicted", "NotMentioned": "Not mentioned"},
        ),
    }
    assert are_questions_independent(long_state, q_independent) is True

    # 3. Independent domain-keyed device state verification over long state -> True
    q_devices = {
        "q_dev_1": ChoiceQuestion(
            instructions="Is lounge_plug in the target set?",
            criteria={"k_no": "no, device is not targeted", "k_yes": "yes, device is targeted"},
        ),
        "q_dev_2": ChoiceQuestion(
            instructions="Is kitchen_light in the target set?",
            criteria={"k_no": "no, device is not targeted", "k_yes": "yes, device is targeted"},
        ),
    }
    assert are_questions_independent(long_state, q_devices) is True

    # 4. BFCL-style tool-call routing -> False
    q_bfcl = {
        "tool_Events_FindEvents": ChoiceQuestion(
            instructions="Given the conversation and function definition, decide whether to call FindEvents.",
            criteria={"yes": "Invoke this tool.", "no": "Do not invoke this tool."},
        ),
        "tool_Movies_FindMovies": ChoiceQuestion(
            instructions="Given the conversation and function definition, decide whether to call FindMovies.",
            criteria={"yes": "Invoke this tool.", "no": "Do not invoke this tool."},
        ),
    }
    assert are_questions_independent(long_state, q_bfcl) is False

    # 5. Cross-question reference / sequential chain -> False
    q_cross_ref = {
        "step_alpha": ChoiceQuestion(
            instructions="Determine the primary category.",
            criteria={"A": "Cat A", "B": "Cat B", "C": "Cat C"},
        ),
        "step_beta": ChoiceQuestion(
            instructions="Given the answer to step_alpha, select the subcategory.",
            criteria={"X": "Sub X", "Y": "Sub Y", "Z": "Sub Z"},
        ),
    }
    assert are_questions_independent(long_state, q_cross_ref) is False

    # 6. Coupled generic yes/no multi-label decomposition -> False
    q_multilabel = {
        "option_0": ChoiceQuestion(
            instructions="Does the text include concept A?", criteria={"yes": "Yes", "no": "No"}
        ),
        "option_1": ChoiceQuestion(
            instructions="Does the text include concept B?", criteria={"yes": "Yes", "no": "No"}
        ),
    }
    assert are_questions_independent(long_state, q_multilabel) is False

    # 7. Structured candidate payload in binary retrieval questions -> False
    q_candidate_dict = {
        "doc_0": ChoiceQuestion(
            instructions={"candidate": "def foo(): pass"},
            criteria={"no": "Not relevant", "yes": "Relevant"},
        ),
        "doc_1": ChoiceQuestion(
            instructions={"candidate": "def bar(): pass"},
            criteria={"no": "Not relevant", "yes": "Relevant"},
        ),
    }
    assert are_questions_independent(long_state, q_candidate_dict) is False


def test_qwen35_4d_mask_wrapping_and_kv_branched_evaluation() -> None:
    tok = _build_test_tokenizer()
    builder = ClassOnePromptBuilder(tok)

    class _DummyHFBackbone(torch.nn.Module):
        def __init__(self, hidden_size: int = 32) -> None:
            super().__init__()
            self.config = SimpleNamespace(model_type="qwen3_5_text", hidden_size=hidden_size)
            self.embed = torch.nn.Embedding(len(tok), hidden_size)
            self.calls: list[dict] = []

        def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask=None,
            position_ids=None,
            past_key_values=None,
            use_cache: bool = False,
            **kwargs,
        ):
            self.calls.append(
                {
                    "input_shape": tuple(input_ids.shape),
                    "attention_mask": attention_mask,
                    "position_ids": position_ids,
                    "has_pkv": past_key_values is not None,
                    "use_cache": use_cache,
                }
            )
            h = self.embed(input_ids)
            pkv = {"cached_prefix_len": input_ids.shape[1]} if use_cache else None
            return SimpleNamespace(last_hidden_state=h, past_key_values=pkv)

    backbone = _DummyHFBackbone(hidden_size=32)
    config = ClassOneConfig(hidden_size=32, head_hidden_size=16)
    model = ClassOneModel(config=config, backbone=backbone)

    state = "Shared contract state text that is significantly longer than the short question instructions below." * 3
    questions = {
        "q1": ChoiceQuestion(
            instructions="Check clause 1.",
            criteria={"Entailment": "Entailed", "Contradiction": "Contradicted", "Neutral": "Neutral"},
        ),
        "q2": ChoiceQuestion(
            instructions="Check clause 2.",
            criteria={"Entailment": "Entailed", "Contradiction": "Contradicted", "Neutral": "Neutral"},
        ),
        "q3": ScoreQuestion(
            instructions="Rate clarity.",
            criteria=["Poor", "Average", "Excellent"],
        ),
        "q4": NoulQuestion(
            instructions="Is signed?",
        ),
    }

    answers, total_tokens = model.evaluate_request(builder=builder, state=state, questions=questions)
    assert set(answers.keys()) == {"q1", "q2", "q3", "q4"}
    assert total_tokens > 0
    # 1 prefix cache call + 4 branched suffix calls = 5 calls
    assert len(backbone.calls) == 5
    assert backbone.calls[0]["use_cache"] is True
    assert backbone.calls[0]["has_pkv"] is False

    # Branched choice/score calls on qwen3_5 should wrap the 4D suffix mask in a dict
    for call in backbone.calls[1:4]:
        assert call["has_pkv"] is True
        assert isinstance(call["attention_mask"], dict)
        assert "full_attention" in call["attention_mask"]
        assert call["attention_mask"]["full_attention"].ndim == 4
        assert call["position_ids"] is not None

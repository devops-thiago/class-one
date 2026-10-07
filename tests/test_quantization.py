"""Tests verifying 8-bit and 4-bit quantization support in ClassOne."""

import pytest
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

from classone.modeling.modeling_classone import ClassOneModel
from classone.schemas import ChoiceQuestion, NoulQuestion, ScoreQuestion
from classone.tokenizer import ClassOnePromptBuilder

REPO_ID = "devops-thiago/classone-gemma4-e2b"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for bitsandbytes quantization")
@pytest.mark.parametrize("quant_mode", ["4bit", "8bit"])
def test_quantized_model_inference_cuda(quant_mode):
    """Verifies that 4-bit and 8-bit quantized models load and evaluate decisions correctly on CUDA."""
    device = "cuda:0"
    tokenizer = AutoTokenizer.from_pretrained(REPO_ID)
    builder = ClassOnePromptBuilder(tokenizer)

    model = ClassOneModel.from_backbone(
        base_model_name_or_path=REPO_ID,
        tokenizer=tokenizer,
        device=device,
        quantization=quant_mode,
    )

    heads_path = hf_hub_download(REPO_ID, "classone_heads.pt")
    heads = torch.load(heads_path, map_location=device)
    model.noul_head.load_state_dict(heads["noul_head"])
    model.choice_head.load_state_dict(heads["choice_head"])
    model.score_head.load_state_dict(heads["score_head"])
    model.eval()

    state = {
        "customer": "Alex",
        "message": "Double charge on transaction #123. Please refund.",
    }
    questions = {
        "refund": NoulQuestion(instructions="Is user requesting a refund?"),
        "dept": ChoiceQuestion(
            instructions="Routing queue:",
            criteria={"billing": "Charges & refunds", "tech": "App bugs"},
        ),
        "urgency": ScoreQuestion(
            instructions="Urgency level:",
            criteria=["low", "medium", "critical"],
        ),
    }

    packed = builder.pack(state=state, questions=questions)

    with torch.no_grad():
        answers = model.evaluate_packed(packed)

    assert "refund" in answers
    assert answers["refund"].type == "noul"
    assert 0.0 <= answers["refund"].noul <= 1.0

    assert "dept" in answers
    assert answers["dept"].type == "choice"
    assert answers["dept"].choice in ["billing", "tech"]
    assert 0.0 <= answers["dept"].confidence <= 1.0

    assert "urgency" in answers
    assert answers["urgency"].type == "score"
    assert 1.0 <= answers["urgency"].score <= 3.0

    del model

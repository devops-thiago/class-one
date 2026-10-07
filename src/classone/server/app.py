"""FastAPI Server for ClassOne System 1 decision models.

Exposes the POST /v1/decide and POST /v1/classone endpoints.
"""

from __future__ import annotations

import os
import threading
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, status
from transformers import AutoTokenizer

from classone.modeling.configuration_classone import ClassOneConfig
from classone.modeling.modeling_classone import ClassOneModel
from classone.schemas import DecisionRequest, DecisionResponse, DecisionUsage
from classone.tokenizer import ClassOnePromptBuilder

# Global thread-safe state
_pipeline_lock = threading.Lock()
_model: ClassOneModel | None = None
_tokenizer: Any | None = None
_prompt_builder: ClassOnePromptBuilder | None = None

MAX_QUESTIONS_PER_REQUEST = 128
MAX_TOKENS_PER_REQUEST = 8192


def get_pipeline() -> tuple[ClassOneModel, ClassOnePromptBuilder]:
    """Lazily and thread-safely initializes or retrieves model and tokenizer pipeline."""
    global _model, _tokenizer, _prompt_builder
    if _model is None:
        with _pipeline_lock:
            if _model is None:
                model_name = os.environ.get("CLASSONE_BASE_MODEL") or "standalone"
                if model_name == "standalone":
                    # Lightweight in-memory standalone mode
                    from tokenizers import Tokenizer
                    from tokenizers.models import WordLevel
                    from tokenizers.pre_tokenizers import Whitespace
                    from transformers import PreTrainedTokenizerFast

                    vocab = {"[UNK]": 0, "[PAD]": 1, "[CLS]": 2, "[SEP]": 3, "[MASK]": 4}
                    tok = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
                    tok.pre_tokenizer = Whitespace()
                    _tokenizer = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]", pad_token="[PAD]")
                    config = ClassOneConfig(hidden_size=256, head_hidden_size=128)
                    _model = ClassOneModel(config)
                else:
                    import torch

                    quant = os.environ.get("CLASSONE_QUANTIZATION")
                    device = os.environ.get("CLASSONE_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")

                    q8_pt_path = os.environ.get("CLASSONE_Q8_MODEL_PATH")
                    if not q8_pt_path and os.path.isfile(model_name) and model_name.endswith(".pt"):
                        q8_pt_path = model_name

                    tok_path = os.environ.get("CLASSONE_TOKENIZER_PATH") or (
                        os.path.dirname(q8_pt_path) if q8_pt_path and not os.path.isdir(model_name) else model_name
                    )
                    _tokenizer = AutoTokenizer.from_pretrained(tok_path)

                    if q8_pt_path and os.path.exists(q8_pt_path) and device == "cpu":
                        print(f"Loading pre-quantized Q8 INT8 model with mmap=True from {q8_pt_path}...")
                        _model = torch.load(q8_pt_path, map_location="cpu", weights_only=False, mmap=True)
                    else:
                        _model = ClassOneModel.from_backbone(
                            model_name,
                            tokenizer=_tokenizer,
                            device=device,
                            quantization=quant,
                        )
                    # Load trained decision heads if available
                    heads_path = os.environ.get("CLASSONE_HEADS_PATH")
                    if not heads_path and os.path.exists(os.path.join(model_name, "classone_heads.pt")):
                        heads_path = os.path.join(model_name, "classone_heads.pt")
                    if not heads_path and "/" in model_name:
                        try:
                            from huggingface_hub import hf_hub_download

                            heads_path = hf_hub_download(model_name, "classone_heads.pt")
                        except Exception:
                            heads_path = None
                    if heads_path and os.path.exists(heads_path):
                        heads = torch.load(heads_path, map_location=_model.device)
                        _model.noul_head.load_state_dict(heads["noul_head"])
                        _model.choice_head.load_state_dict(heads["choice_head"])
                        _model.score_head.load_state_dict(heads["score_head"])
                        if str(_model.device) == "cpu":
                            _model.noul_head.to(dtype=torch.float32)
                            _model.choice_head.to(dtype=torch.float32)
                            _model.score_head.to(dtype=torch.float32)

                _prompt_builder = ClassOnePromptBuilder(_tokenizer)

    return _model, _prompt_builder


def set_pipeline(model: ClassOneModel, prompt_builder: ClassOnePromptBuilder):
    """Allows programmatic injection of model and tokenizer (e.g., during tests)."""
    global _model, _prompt_builder
    with _pipeline_lock:
        _model = model
        _prompt_builder = prompt_builder


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle hook warming up model pipeline at server start."""
    get_pipeline()
    yield


app = FastAPI(
    title="ClassOne System 1 API",
    description="Open-source System 1 decision engine built on Gemma 4 E2B",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health_check():
    return {"status": "ok", "service": "classone-system-1"}


@app.post("/v1/decide", response_model=DecisionResponse)
@app.post("/v1/classone", response_model=DecisionResponse)
@app.post("/v1/systemone", response_model=DecisionResponse)
def evaluate_decisions(request: DecisionRequest) -> DecisionResponse:
    """Evaluates arbitrary typed questions against unstructured state in a single pass."""
    # Input validation guards against DoS
    if len(request.questions) > MAX_QUESTIONS_PER_REQUEST:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Too many questions: maximum allowed is {MAX_QUESTIONS_PER_REQUEST}.",
        )

    try:
        model, builder = get_pipeline()
        packed = builder.pack(state=request.state, questions=request.questions)

        input_token_count = int(packed.input_ids.shape[1])
        if input_token_count > MAX_TOKENS_PER_REQUEST:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Prompt length ({input_token_count} tokens) exceeds maximum limit ({MAX_TOKENS_PER_REQUEST}).",
            )

        answers = model.evaluate_packed(packed)

        return DecisionResponse(
            model=request.model,
            answers=answers,
            usage=DecisionUsage(
                input_tokens=input_token_count,
                output_tokens=0,  # System 1 models don't generate token streams
            ),
        )
    except HTTPException:
        raise
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(val_err),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal decision engine error: {exc}",
        )

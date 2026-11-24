"""FastAPI Server for ClassOne System 1 decision models.

Exposes the POST /v1/decide and POST /v1/classone endpoints.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import pathlib
import threading
from collections.abc import Mapping
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, status
from transformers import AutoTokenizer

from classone.modeling.configuration_classone import ClassOneConfig
from classone.modeling.modeling_classone import ClassOneModel
from classone.schemas import DecisionRequest, DecisionResponse, DecisionUsage
from classone.stage2_readout import (
    ChatDecisionPromptBuilder,
    evaluate_stage2_request,
    load_stage2_snapshot,
    strip_unused_multimodal_towers,
)
from classone.tokenizer import ClassOnePromptBuilder

logger = logging.getLogger(__name__)

# Global thread-safe state
_pipeline_lock = threading.Lock()
_model: ClassOneModel | None = None
_tokenizer: Any | None = None
_prompt_builder: ClassOnePromptBuilder | None = None
_chat_builder: Any | None = None
_warned_uncapped = False

MAX_QUESTIONS_PER_REQUEST = 128


def _get_field(obj: Any, name: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, Mapping):
        return obj.get(name)
    return getattr(obj, name, None)


def _positive_int_or_none(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if value > 0 else None


def context_length_from_config(config: Any) -> int | None:
    """Resolve positive max_position_embeddings from a Hugging Face / ClassOne config."""
    if config is None:
        return None
    text_config = _get_field(config, "text_config")
    for source in (text_config, config):
        limit = _positive_int_or_none(_get_field(source, "max_position_embeddings"))
        if limit is not None:
            return limit
    return None


def context_length_from_model(model: Any) -> int | None:
    """Resolve context window from model.config or model.backbone.config."""
    if model is None:
        return None
    limit = context_length_from_config(getattr(model, "config", None))
    if limit is not None:
        return limit
    backbone = getattr(model, "backbone", None)
    if backbone is not None:
        return context_length_from_config(getattr(backbone, "config", None))
    return None


def get_max_tokens_per_request(model: Any | None = None) -> int | None:
    """Resolve the per-request prompt token limit dynamically.

    Priority:
    1. CLASSONE_MAX_TOKENS_PER_REQUEST env var (if non-empty positive int)
    2. Loaded model / backbone config max_position_embeddings (preferring text_config)
    3. Uncapped (None) in standalone mode when no model window is configured
    """
    global _warned_uncapped
    raw = os.environ.get("CLASSONE_MAX_TOKENS_PER_REQUEST")
    if raw is not None and raw.strip():
        limit = int(raw)
        if limit <= 0:
            raise ValueError("CLASSONE_MAX_TOKENS_PER_REQUEST must be a positive integer.")
        return limit

    target_model = model if model is not None else _model
    if target_model is None:
        target_model, _ = get_pipeline()

    limit = context_length_from_model(target_model)
    if limit is None and not _warned_uncapped:
        logger.warning(
            "Loaded model config does not declare max_position_embeddings; leaving per-request prompt length "
            "uncapped. Set CLASSONE_MAX_TOKENS_PER_REQUEST to enforce a VRAM or latency budget."
        )
        _warned_uncapped = True
    return limit


def _resolve_codebook_path(
    model_name: str,
    cfg: Mapping[str, Any],
    revision: str | None,
    offline_only: bool,
) -> str | None:
    """Resolves a bundled codebook_file from a local standalone dir or Hub repo."""
    cb_file = str(cfg.get("codebook_file") or "").strip() if isinstance(cfg, Mapping) else ""
    if not cb_file:
        return None
    if os.path.isdir(model_name):
        cand = pathlib.Path(model_name) / cb_file
        return str(cand) if cand.exists() else None
    if "/" in model_name:
        try:
            from huggingface_hub import hf_hub_download

            return hf_hub_download(model_name, cb_file, revision=revision, local_files_only=offline_only)
        except Exception:
            return None
    return None


def get_pipeline() -> tuple[ClassOneModel, ClassOnePromptBuilder]:
    """Lazily and thread-safely initializes or retrieves model and tokenizer pipeline."""
    global _model, _tokenizer, _prompt_builder, _chat_builder
    if _model is None:
        with _pipeline_lock:
            if _model is None:
                model_name = os.environ.get("CLASSONE_BASE_MODEL") or "standalone"
                snapshot_path = os.environ.get("CLASSONE_SNAPSHOT_PATH")
                if model_name == "standalone" and not snapshot_path:
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
                    _prompt_builder = ClassOnePromptBuilder(_tokenizer)
                elif snapshot_path:
                    if model_name == "standalone":
                        meta_file = pathlib.Path(snapshot_path) / "snapshot_metadata.json"
                        if meta_file.exists():
                            model_name = json.loads(meta_file.read_text(encoding="utf-8")).get("base_model", model_name)
                    import torch

                    quant = os.environ.get("CLASSONE_QUANTIZATION")
                    device = os.environ.get("CLASSONE_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
                    _model, _tokenizer, _chat_builder = load_stage2_snapshot(
                        model_name,
                        snapshot_path,
                        device=device,
                        quantization=quant,
                    )
                    _prompt_builder = ClassOnePromptBuilder(copy.deepcopy(_tokenizer))
                else:
                    import torch

                    quant = os.environ.get("CLASSONE_QUANTIZATION")
                    device = os.environ.get("CLASSONE_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")

                    q8_pt_path = os.environ.get("CLASSONE_Q8_MODEL_PATH")
                    if not q8_pt_path and os.path.isfile(model_name) and model_name.endswith(".pt"):
                        q8_pt_path = model_name

                    revision = os.environ.get("CLASSONE_REVISION", "").strip() or None
                    tok_path = os.environ.get("CLASSONE_TOKENIZER_PATH") or (
                        os.path.dirname(q8_pt_path) if q8_pt_path and not os.path.isdir(model_name) else model_name
                    )
                    eval_mode_env = os.getenv("CLASSONE_EVAL_MODE", "").strip().lower()
                    offline_only = os.environ.get("HF_HUB_OFFLINE") == "1"
                    disk_cfg: dict[str, Any] = {}
                    if os.path.isdir(model_name):
                        cfg_file = pathlib.Path(model_name) / "config.json"
                        if cfg_file.exists():
                            disk_cfg = json.loads(cfg_file.read_text(encoding="utf-8")).get("classone_config", {}) or {}
                    elif "/" in model_name and not q8_pt_path:
                        try:
                            from huggingface_hub import hf_hub_download

                            cfg_file_hub = hf_hub_download(
                                model_name,
                                "config.json",
                                revision=revision,
                                local_files_only=offline_only,
                            )
                            disk_cfg = (
                                json.loads(pathlib.Path(cfg_file_hub).read_text(encoding="utf-8")).get(
                                    "classone_config", {}
                                )
                                or {}
                            )
                        except Exception:
                            disk_cfg = {}
                    is_stage2_cfg = eval_mode_env == "stage2" or disk_cfg.get("eval_mode") == "stage2"

                    local_only = os.path.isdir(tok_path) or offline_only
                    tok_kwargs: dict[str, Any] = {
                        "trust_remote_code": True,
                        "local_files_only": local_only,
                    }
                    if revision and not os.path.isdir(tok_path):
                        tok_kwargs["revision"] = revision
                    _tokenizer = AutoTokenizer.from_pretrained(tok_path, **tok_kwargs)
                    cb_path = _resolve_codebook_path(model_name, disk_cfg, revision, offline_only)
                    if is_stage2_cfg:
                        _chat_builder = ChatDecisionPromptBuilder(copy.deepcopy(_tokenizer), codebook_path=cb_path)
                    elif getattr(_tokenizer, "chat_template", None):
                        try:
                            _chat_builder = ChatDecisionPromptBuilder(copy.deepcopy(_tokenizer), codebook_path=cb_path)
                        except Exception:
                            _chat_builder = None

                    if q8_pt_path and os.path.exists(q8_pt_path) and device == "cpu":
                        print(f"Loading pre-quantized Q8 INT8 model with mmap=True from {q8_pt_path}...")
                        _model = torch.load(q8_pt_path, map_location="cpu", weights_only=False, mmap=True)
                    else:
                        backbone_dtype = torch.bfloat16 if (is_stage2_cfg and device != "cpu" and not quant) else None
                        bb_kwargs: dict[str, Any] = {}
                        if revision and not os.path.isdir(model_name):
                            bb_kwargs["revision"] = revision
                        _model = ClassOneModel.from_backbone(
                            model_name,
                            tokenizer=_tokenizer,
                            device=device,
                            quantization=quant,
                            torch_dtype=backbone_dtype,
                            **bb_kwargs,
                        )
                    # Load trained decision heads if available
                    heads_path = os.environ.get("CLASSONE_HEADS_PATH")
                    if not heads_path and os.path.exists(os.path.join(model_name, "classone_heads.pt")):
                        heads_path = os.path.join(model_name, "classone_heads.pt")
                    if not heads_path and "/" in model_name:
                        try:
                            from huggingface_hub import hf_hub_download

                            heads_path = hf_hub_download(
                                model_name,
                                "classone_heads.pt",
                                revision=revision,
                                local_files_only=offline_only,
                            )
                        except Exception:
                            heads_path = None
                    if heads_path and os.path.exists(heads_path):
                        heads = torch.load(heads_path, map_location=_model.device, weights_only=True)
                        _model.noul_head.load_state_dict(heads["noul_head"], strict=False)
                        _model.choice_head.load_state_dict(heads["choice_head"], strict=False)
                        _model.score_head.load_state_dict(heads["score_head"], strict=False)
                        if str(_model.device) == "cpu":
                            _model.noul_head.to(dtype=torch.float32)
                            _model.choice_head.to(dtype=torch.float32)
                            _model.score_head.to(dtype=torch.float32)

                    backbone_cfg = getattr(getattr(_model, "backbone", None), "config", None)
                    runtime_cls_cfg = (
                        getattr(backbone_cfg, "classone_config", None)
                        or getattr(getattr(_model, "config", None), "classone_config", None)
                        or {}
                    )
                    stage2_active = is_stage2_cfg or (runtime_cls_cfg.get("eval_mode") == "stage2")
                    if stage2_active and _chat_builder is None:
                        cb_path = _resolve_codebook_path(
                            model_name,
                            disk_cfg or runtime_cls_cfg,
                            revision,
                            offline_only,
                        )
                        _chat_builder = ChatDecisionPromptBuilder(copy.deepcopy(_tokenizer), codebook_path=cb_path)
                    if stage2_active and getattr(_model, "backbone", None) is not None:
                        strip_unused_multimodal_towers(_model.backbone, clear_cuda_cache=torch.cuda.is_available())

                    _prompt_builder = ClassOnePromptBuilder(_tokenizer)

    return _model, _prompt_builder


def set_pipeline(model: ClassOneModel, prompt_builder: ClassOnePromptBuilder, chat_builder: Any | None = None):
    """Allows programmatic injection of model and tokenizer (e.g., during tests)."""
    global _model, _prompt_builder, _chat_builder
    with _pipeline_lock:
        _model = model
        _prompt_builder = prompt_builder
        _chat_builder = chat_builder


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
        max_tokens = get_max_tokens_per_request(model)
        eval_mode = os.getenv("CLASSONE_EVAL_MODE", "stage1").strip().lower()
        cls_cfg = getattr(getattr(model, "backbone", None), "config", None)
        stage2_cfg = (
            getattr(cls_cfg, "classone_config", None)
            or getattr(getattr(model, "config", None), "classone_config", None)
            or {}
        )
        use_stage2 = _chat_builder is not None and (
            eval_mode == "stage2"
            or (
                eval_mode not in ("legacy_packed", "c2_packed_only")
                and (
                    bool(os.environ.get("CLASSONE_SNAPSHOT_PATH", "").strip())
                    or stage2_cfg.get("eval_mode") == "stage2"
                )
            )
        )

        if use_stage2:
            m2_k_threshold = int(os.getenv("CLASSONE_M2_K_THRESHOLD", str(stage2_cfg.get("m2_k_threshold", 64))))
            m2_top_k = int(os.getenv("CLASSONE_M2_TOP_K", str(stage2_cfg.get("m2_top_k", 16))))
            answers, input_token_count = evaluate_stage2_request(
                model,
                _chat_builder,
                state=request.state,
                questions=request.questions,
                m2_k_threshold=m2_k_threshold,
                m2_top_k=m2_top_k,
                max_tokens=max_tokens or 4096,
            )
        elif eval_mode == "legacy_packed":
            packed = builder.pack(state=request.state, questions=request.questions, options_in_prefix=False)
            input_token_count = int(packed.input_ids.shape[1])
            if max_tokens is not None and input_token_count > max_tokens:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Prompt length ({input_token_count} tokens) exceeds maximum context length ({max_tokens}).",
                )
            answers = model.evaluate_packed(packed, position_invariant=False)
        elif hasattr(model, "evaluate_request"):
            answers, input_token_count = model.evaluate_request(
                builder=builder,
                state=request.state,
                questions=request.questions,
                options_in_prefix=True,
                position_invariant=True,
                adaptive_branching=(eval_mode != "c2_packed_only"),
                max_tokens=max_tokens,
            )
        else:
            packed = builder.pack(state=request.state, questions=request.questions)
            input_token_count = int(packed.input_ids.shape[1])
            if max_tokens is not None and input_token_count > max_tokens:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Prompt length ({input_token_count} tokens) exceeds maximum context length ({max_tokens}).",
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

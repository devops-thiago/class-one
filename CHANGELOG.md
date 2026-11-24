# Changelog

All notable changes to `classone` are documented in this file.

## [0.4.0] - 2026-11-24

### Added
- **Gemma 4 Stage 2 LM-head readout (`classone.stage2_readout`, `classone.modeling`):**
  - Added versioned Gemma 4 single-token round-trip codebook (`src/classone/codebooks/gemma4_e2b_codebook_v1.json`) and tied embedding slice readout (`CodebookLMHead`) with `final_logit_softcapping` support (`30.0`).
  - Preserved FP32 RoPE `inv_freq` buffers and FP32 decision heads (`noul_head`, `choice_head`, `score_head`) under BF16 backbone loading, plus unused multimodal tower stripping (`strip_unused_multimodal_towers`).
  - Added standalone Stage 2 export support for unmodified base weights (`scripts/export_stage2_merged.py --allow-no-lora --strip-multimodal`) used by `devops-thiago/classone-gemma4-e2b` v1.1 (`2ba24ff`, Decision Index 0.3 public `24.64`, `apolinario/decision-index#520`).
- **2-GPU model-parallel Stage 2 training support (`scripts/train_stage2_lm_letter.py`, `classone.modeling`):**
  - Added `--model-parallel` / `--mp-split-layer` sharding across 2 GPUs for Qwen 3.5 9B and Gemma 4 E4B, including `UserDict` `shared_kv_states` device migration, `--resume-lora` device guards, and FP32 `inv_freq` preservation. *(Stage 2 candidates for 9B and E4B were rejected by preregistered gates — `REJECT_ALL` / `REJECT_UNSEEN` — leaving published 9B/E4B checkpoints unchanged.)*

### Changed
- Added light/dark ClassOne logo assets (`assets/classone-logo.png`, `assets/classone-logo-dark.png`) and centered header in `README.md`.
- Normalized repository line endings to LF via `.gitattributes` and updated live Gemma 4 tokenizer cache lookup in `tests/test_stage2_readout.py` to use `huggingface_hub.try_to_load_from_cache`.

## [0.3.0] - 2026-11-18

### Added
- **Stage 2 LM-head letter readout & M2 shortlist routing (`classone.stage2_readout`):**
  - Exported `ChatDecisionPromptBuilder`, `ChatPackedQuestion`, `build_single_token_labels`, `compute_lm_letter_logits`, `evaluate_stage2_request`, and `load_stage2_snapshot` from top-level `classone`.
  - Added single-token LM-head choice readout with automatic two-pass M2 top-`k` shortlist routing (`m2_k_threshold=64`, `m2_top_k=16`).
- **Stage 2 configuration propagation (`ClassOneConfig` & `ClassOneModel`):**
  - Added `eval_mode`, `m2_k_threshold`, `m2_top_k`, and `classone_config` fields to `ClassOneConfig`, populated automatically from `backbone.config.classone_config` inside `ClassOneModel.from_backbone`.
  - Defaulted non-quantized Stage 2 backbones on accelerators to `torch.bfloat16` when `torch_dtype` is omitted.
  - Extended `ClassOneModel.evaluate_request` to dispatch Stage 2 / `ChatDecisionPromptBuilder` requests with `eval_mode`, `m2_k_threshold`, and `m2_top_k` overrides.
- **FastAPI server Hub & revision support (`classone.server.app`):**
  - Added `CLASSONE_REVISION` support and pre-load `config.json` Hub inspection so Stage 2 Hub models (such as `devops-thiago/classone-qwen3.5-4b` at tag `v1.1`) auto-configure Stage 2 BF16 loading and `classone_heads.pt` downloads.
- **Stage 2 `v1.1` Qwen 3.5 4B baseline sync (`v1_1_merge_0.40_0.60`):**
  - Updated registered `devops-thiago/classone-qwen3.5-4b` `classone_heads.pt` SHA256 to `147AFEBDF277CF66375B18590BFD1C305F09D3DD1354D7F83EC7A346754C1E02` across `benchmarks/baselines.json` and `README.md`.

### Changed
- Added Stage 2 LoRA trainer (`scripts/train_stage2_lm_letter.py`), standalone exporter (`scripts/export_stage2_merged.py`), `v1.1` When2Call micro-stage builder/evaluator (`scripts/build_w2c_micro_corpus.py`, `scripts/eval_v1_1_candidate.py`), and unit test suite (`tests/test_stage2_readout.py`).

## [0.2.4] - 2026-11-13
- Added CI verification (`scripts/verify_hub_heads_sha.py`) and non-regression publication gates (`src/classone/publishing.py`).
- Updated launch README and Hugging Face Hub model card provenance metadata.

## [0.2.3] - 2026-11-12
- Derived server default prompt token limit from `model.config` / `text_config.max_position_embeddings`.

## [0.2.2] - 2026-11-12
- Normalized rounded probability sums to `1.0`, honored `CLASSONE_MAX_TOKENS_PER_REQUEST`, coerced non-string instructions/criteria, and declared `httpx` runtime dependency.

## [0.2.1] - 2026-11-11
- Promoted Qwen 3.5 9B Pillar 2 checkpoints and aligned multi-suite scorecards.

## [0.2.0] - 2026-11-10
- Initial public PyPI SDK (`ClassOneClient`, `AsyncClassOneClient`) and standalone checkpoint release workflow.

# Special Delimiter Tokens & Gemma 4 Per-Layer Embeddings (PLE)

## The Core Concept

ClassOne introduces custom delimiter tokens that encapsulate different portions of the decision sequence:
- `<|state_start|>` and `<|state_end|>`: Frame the primary state, context, or document.
- `<|noul_start|>` and `<|noul_end|>`: Frame refusal / binary guardrail decision criteria.
- `<|choice_start|>` and `<|choice_end|>`: Frame the choice question instructions.
- `<|opt_start|>` and `<|opt_end|>`: Frame each candidate option description.
- `<|score_start|>` and `<|score_end|>`: Frame ordinal rubric instructions.
- `<|level_start|>` and `<|level_end|>`: Frame each rubric grade level.
- `<|pause|>`: Latent thinking tokens placed before the decision query token to provide computational sequence depth.

---

## Tokenizer Serialization Rule

When saving a merged model or pushing to Hugging Face Hub, the tokenizer must be serialized via `ClassOnePromptBuilder`:

```python
from transformers import AutoTokenizer
from classone.tokenizer import ClassOnePromptBuilder

tok = AutoTokenizer.from_pretrained(target_dir)
builder = ClassOnePromptBuilder(tok)
tok.save_pretrained(target_dir)
```

Without this step, `tokenizer.json` will lack the custom delimiters, and raw strings like `<|state_start|>` will be broken into subwords (`<`, `|`, `state`, `_`, `start`, `|>`), completely destroying the attention representation learned during training.

---

## Gemma 4 Architecture: The Per-Layer Embeddings (PLE) Gotcha

Gemma 4 models (e.g. `google/gemma-4-E4B-it`, `google/gemma-4-2B`) use **Per-Layer Embeddings (PLE)** across all 42 transformer layers.

### Why You MUST NOT Call `tokenizer.add_special_tokens()` on Gemma 4:
1. Resizing embedding matrices on Gemma 4 requires resizing 42 distinct embedding projections across all layers.
2. In memory, this instantly expands VRAM by **5.25 GB**, causing CUDA OOM on 16 GB GPUs.
3. If token weights are merged without updating all 42 PLE layers, an architectural mismatch crash occurs during forward pass.

### The Solution: In-Vocab `<unused*>` Mapping
Gemma's vocabulary provides pre-allocated, unused token IDs (`<unused0>` through `<unused12>`).
`ClassOnePromptBuilder` automatically maps ClassOne delimiters to these unused tokens on Gemma 4 backbones:
- `<|state_start|>` -> `<unused0>`
- `<|state_end|>`   -> `<unused1`>
- `<|pause|>`       -> `<unused12>`

This completely avoids resizing vocabulary or modifying PLE layers, guaranteeing zero extra VRAM and 100% architectural stability.

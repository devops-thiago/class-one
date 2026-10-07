import torch
from transformers import AutoTokenizer

from classone.modeling.modeling_classone import ClassOneModel
from classone.schemas import ChoiceQuestion, NoulQuestion, ScoreQuestion
from classone.tokenizer import ClassOnePromptBuilder

device = "cuda:0"
model_id = "google/gemma-4-12B-it"
print(f"[*] Probing ClassOne initialization with {model_id} (4-bit NF4)...")

tok = AutoTokenizer.from_pretrained(model_id)
builder = ClassOnePromptBuilder(tok)
mapped = builder.delimiter_map["<|state_start|>"]
print(f"[✓] Tokenizer loaded! Delimiters mapped to: {mapped}")

model = ClassOneModel.from_backbone(
    base_model_name_or_path=model_id,
    tokenizer=tok,
    device=device,
    quantization="4bit",
)

vram_alloc = torch.cuda.memory_allocated(0) / (1024**3)
print(f"[✓] 12B Model successfully loaded in 4-bit NF4 on {device}!")
print(f"  Allocated VRAM: {vram_alloc:.2f} GB (Leaves {16.0 - vram_alloc:.2f} GB free headroom on GPU 0!)")
print(f"  ChoiceHead in_features: {model.choice_head.q_proj[0].in_features}")
print(f"  NoulHead in_features  : {model.noul_head.net[0].in_features}")

# Test forward decision pass
state = "Enterprise MSA Agreement: Liability is capped at 12 months fees ($1.2M), excluding IP indemnity."
questions = {
    "liability": ChoiceQuestion(
        instructions="Determine governing liability cap:",
        criteria={"capped_12mo": "Capped at 12 months fees ($1.2M)", "uncapped_ip": "Uncapped for IP indemnity"},
    ),
    "ip_capped": NoulQuestion(instructions="Is IP indemnity subject to the 12-month liability cap?"),
    "exposure": ScoreQuestion(
        instructions="Rate legal exposure severity:", criteria=["routine", "elevated", "material_breach"]
    ),
}

packed = builder.pack(state, questions)

with torch.no_grad():
    answers = model.evaluate_packed(packed)

print("[✓] evaluate_packed executed successfully on Gemma 4 12B!")
for k, v in answers.items():
    ans_str = str(getattr(v, "choice", getattr(v, "noul", getattr(v, "score", ""))))
    print(f"  • {k:10s}: {ans_str}")

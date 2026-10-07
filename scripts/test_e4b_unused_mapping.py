import torch
from transformers import AutoTokenizer, BitsAndBytesConfig

from classone.modeling.modeling_classone import ClassOneModel

device = "cuda:0"
model_id = "google/gemma-4-E4B-it"
print("Testing Gemma 4 E4B with in-vocab <unusedN> delimiter mapping...")

tok = AutoTokenizer.from_pretrained(model_id)

delimiters = [
    "<|state_start|>",
    "<|state_end|>",
    "<|noul_start|>",
    "<|noul_end|>",
    "<|choice_start|>",
    "<|choice_end|>",
    "<|opt_start|>",
    "<|opt_end|>",
    "<|score_start|>",
    "<|score_end|>",
    "<|level_start|>",
    "<|level_end|>",
]
delim_map = {d: f"<unused{i}>" for i, d in enumerate(delimiters)}
for d, u in delim_map.items():
    uid = tok.convert_tokens_to_ids(u)
    print(f"  {d:18s} -> {u:10s} (ID: {uid})")

# Load 4-bit model without resizing embeddings
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = ClassOneModel.from_backbone(
    base_model_name_or_path=model_id,
    tokenizer=None,  # DO NOT RESIZE
    device=device,
    quantization="4bit",
)

print("[✓] Model loaded without resizing token embeddings!")

# Construct prompt text using mapped delimiters
prompt_text = (
    f"{delim_map['<|state_start|>']}\nCompany T&E Policy: Lodging capped at $350/night under Endorsement L-2.\n{delim_map['<|state_end|>']}\n"
    f"{delim_map['<|choice_start|>']}\nDetermine settlement decision:\n"
    f"{delim_map['<|opt_start|>']}\npay_sublimit: Payment capped at $350 sublimit\n{delim_map['<|opt_end|>']}\n"
    f"{delim_map['<|opt_start|>']}\npay_full: Pay full invoice without limit\n{delim_map['<|opt_end|>']}\n"
    f"{delim_map['<|choice_end|>']}"
)

encoded = tok(prompt_text, return_tensors="pt").to(device)
print(
    f"Prompt tokenized! Shape: {encoded.input_ids.shape}, max ID: {encoded.input_ids.max().item()} (< {tok.vocab_size})"
)

with torch.no_grad():
    hidden = model.extract_hidden_states(encoded.input_ids, encoded.attention_mask)

print(f"[✓] Forward pass successful! Hidden shape: {hidden.shape}")

# Test ChoiceHead
q_vec = hidden[0, -1]
opt_vecs = hidden[0, [15, 30]]  # sample positions
probs = model.choice_head(q_vec, opt_vecs)
print(f"[✓] ChoiceHead executed successfully on Gemma 4 E4B! Probabilities: {probs}")

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer

from classone.modeling.loss import RLCDLoss
from classone.modeling.modeling_classone import ClassOneModel
from classone.schemas import ChoiceQuestion
from classone.tokenizer import ClassOnePromptBuilder

device = "cuda:0"
model_id = "google/gemma-4-12B-it"
print("[*] Testing complete forward + backward pass on Gemma 4 12B (4-bit NF4 LoRA)...")

tok = AutoTokenizer.from_pretrained(model_id)
builder = ClassOnePromptBuilder(tok)

model = ClassOneModel.from_backbone(
    base_model_name_or_path=model_id,
    tokenizer=tok,
    device=device,
    quantization="4bit",
)

# Freeze base model
for p in model.backbone.parameters():
    p.requires_grad = False

# Attach LoRA adapters
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules=r".*language_model.*(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj).*",
)
model.backbone = get_peft_model(model.backbone, lora_config)
model.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

# Decision heads remain trainable
for head in [model.noul_head, model.choice_head, model.score_head]:
    for p in head.parameters():
        p.requires_grad = True

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"[✓] LoRA adapters attached! Trainable parameters: {trainable_params:,}")

vram_post_lora = torch.cuda.memory_allocated(0) / (1024**3)
print(f"  VRAM with trainable adapters: {vram_post_lora:.2f} GB (Leaves {16.0 - vram_post_lora:.2f} GB free!)")

# Create sample decision item
state = (
    "Commercial Lease Agreement: Section 14 limits damages to 6 months base rent ($180,000). Tenant claims $450,000."
)
q = ChoiceQuestion(
    instructions="Determine damage cap limit:",
    criteria={
        "cap_180k": "Damages capped at 6 months rent ($180,000)",
        "uncapped": "Damages uncapped due to gross negligence",
    },
)
packed = builder.pack(state, {"damage_cap": q})

# Forward pass
input_ids = packed.input_ids.to(device)
attention_mask = packed.attention_mask.to(device)
hidden_states = model.extract_hidden_states(input_ids, attention_mask)
seq_hidden = hidden_states[0]

span = packed.questions["damage_cap"]
h_query = seq_hidden[span.query_token_idx]
h_opts = torch.stack([seq_hidden[idx] for idx in span.option_token_indices])

probs, logits = model.choice_head(h_query, h_opts, return_logits=True)

# Compute loss
loss_fn = RLCDLoss(brier_weight=0.5, log_weight=1.0, focal_gamma=2.0)
target = torch.tensor([0], device=device)  # class index 0 (capped)
loss = loss_fn.forward_multiclass(probs.unsqueeze(0), target, logits=logits.unsqueeze(0))
print(f"[✓] Forward loss computed: {loss.item():.4f}")

# Backward pass
loss.backward()

lora_grads = sum(1 for p in model.backbone.parameters() if p.grad is not None)
head_grads = sum(1 for p in model.choice_head.parameters() if p.grad is not None)

print("[✓] Backward pass successful!")
print(f"  LoRA layers with active gradients : {lora_grads}")
print(f"  ChoiceHead parameters with grads : {head_grads}")

vram_post_backward = torch.cuda.memory_allocated(0) / (1024**3)
vram_peak = torch.cuda.max_memory_allocated(0) / (1024**3)
print(f"  VRAM post-backward: {vram_post_backward:.2f} GB (Total Peak: {vram_peak:.2f} GB)")
print(
    f"[✓] Safe Headroom on GPU 0: {16.0 - vram_peak:.2f} GB free ({(16.0 - vram_peak) / 16.0 * 100:.1f}% free margin)!"
)

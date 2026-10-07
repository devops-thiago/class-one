import torch
from transformers import AutoTokenizer

from classone.modeling.modeling_classone import ClassOneModel
from classone.schemas import ChoiceQuestion, NoulQuestion, ScoreQuestion
from classone.tokenizer import ClassOnePromptBuilder
from classone.train.trainer import ClassOneTrainer, TrainingItem

print("Initializing Dual-GPU ClassOne Training...")

device_map = {
    "vision_tower": 0,
    "audio_tower": 0,
    "embed_vision": 0,
    "embed_audio": 0,
    "language_model.embed_tokens": 0,
    "language_model.rotary_emb": 0,
    "language_model.embed_tokens_per_layer": 0,
    "language_model.per_layer_model_projection": 0,
    "language_model.per_layer_projection_norm": 0,
}
for i in range(18):
    device_map[f"language_model.layers.{i}"] = 0
for i in range(18, 35):
    device_map[f"language_model.layers.{i}"] = 1
device_map["language_model.norm"] = 1

model_id = "devops-thiago/classone-gemma4-e2b"
tokenizer = AutoTokenizer.from_pretrained(model_id)
builder = ClassOnePromptBuilder(tokenizer)

model = ClassOneModel.from_backbone(
    base_model_name_or_path=model_id,
    tokenizer=tokenizer,
    device_map=device_map,
    torch_dtype=torch.bfloat16,
)

trainer = ClassOneTrainer(
    model=model,
    prompt_builder=builder,
    lr=2e-4,
    device="cuda:0",
)

print("[*] Attaching LoRA across both GPUs...")
trainer.enable_lora(r=16, lora_alpha=32)

item = TrainingItem(
    state={"message": "System outage on production payment server."},
    questions={
        "is_outage": NoulQuestion(instructions="Is this an outage?"),
        "dept": ChoiceQuestion(instructions="Route team:", criteria={"infra": "Infrastructure", "sales": "Sales"}),
        "urgency": ScoreQuestion(instructions="Urgency:", criteria=["low", "high"]),
    },
    targets={"is_outage": 1.0, "dept": "infra", "urgency": 2},
)

print("[*] Executing dual-GPU training step...")
stats = trainer.train_step([item])
print(f"[✓] Dual-GPU training step succeeded! Loss: {stats['loss']:.4f}")
print("  VRAM on GPU 0:", round(torch.cuda.memory_allocated(0) / (1024**3), 2), "GB")
print("  VRAM on GPU 1:", round(torch.cuda.memory_allocated(1) / (1024**3), 2), "GB")

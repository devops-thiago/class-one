import torch
from transformers import AutoModel

print("Testing balanced dual-GPU split across 16GB GPUs...")
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
# 35 transformer layers: layers 0-17 on GPU 0, 18-34 on GPU 1
for i in range(18):
    device_map[f"language_model.layers.{i}"] = 0
for i in range(18, 35):
    device_map[f"language_model.layers.{i}"] = 1
device_map["language_model.norm"] = 1

model = AutoModel.from_pretrained(
    "devops-thiago/classone-gemma4-e2b",
    device_map=device_map,
    torch_dtype=torch.bfloat16,
)

print("VRAM on GPU 0:", round(torch.cuda.memory_allocated(0) / (1024**3), 2), "GB")
print("VRAM on GPU 1:", round(torch.cuda.memory_allocated(1) / (1024**3), 2), "GB")
print("Model split cleanly across both GPUs!")

# Test a forward pass across both GPUs!
x = torch.randint(0, 1000, (1, 32), device="cuda:0")
with torch.no_grad():
    out = model(x)
print("Forward pass successful across 2 GPUs! Output shape:", out.last_hidden_state.shape)
print("Output tensor device:", out.last_hidden_state.device)

import math

import torch

heads_path = "checkpoints/classone_gemma4_e2b/classone_heads.pt"
heads = torch.load(heads_path, map_location="cpu")

print("Original temperatures in checkpoint:")
for h_name in ["noul_head", "choice_head", "score_head"]:
    raw_key = "choice_evaluator.temperature_raw" if h_name == "score_head" else "temperature_raw"
    if raw_key in heads[h_name]:
        raw_val = heads[h_name][raw_key].item()
        softplus_val = math.log(1.0 + math.exp(raw_val)) + 1e-4
        print(f"  {h_name}: raw = {raw_val:.4f}, effective temp = {softplus_val:.4f}")

# Target optimal temperatures
targets = {
    "noul_head": 0.50,
    "choice_head": 0.50,
    "score_head": 0.60,
}

print("\nUpdating temperatures to optimal target bounds:")
for h_name, target_t in targets.items():
    raw_key = "choice_evaluator.temperature_raw" if h_name == "score_head" else "temperature_raw"
    if raw_key in heads[h_name]:
        # target_t = softplus(raw) => raw = log(exp(target_t) - 1)
        new_raw = math.log(math.exp(target_t) - 1.0)
        heads[h_name][raw_key] = torch.tensor([new_raw], dtype=heads[h_name][raw_key].dtype)
        print(f"  {h_name}: set effective temp to {target_t:.4f} (raw = {new_raw:.4f})")

torch.save(heads, heads_path)
print(f"[✓] Saved updated temperatures to {heads_path}")

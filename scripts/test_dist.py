import os

import torch
import torch.distributed as dist

local_rank = int(os.environ.get("LOCAL_RANK", 0))
world_size = int(os.environ.get("WORLD_SIZE", 1))

torch.cuda.set_device(local_rank)
dist.init_process_group("gloo")

t = torch.tensor([float(local_rank)], device=f"cuda:{local_rank}")
print(f"Rank {local_rank}/{world_size} initialized on GPU {torch.cuda.current_device()}!")
try:
    dist.all_reduce(t)
    print(f"Rank {local_rank} all_reduce value: {t.item()} (SUCCESS)")
except Exception as e:
    print(f"Rank {local_rank} all_reduce error: {e}")

dist.destroy_process_group()

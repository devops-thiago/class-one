import os

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

local_rank = int(os.environ.get("LOCAL_RANK", 0))
world_size = int(os.environ.get("WORLD_SIZE", 1))

torch.cuda.set_device(local_rank)
dist.init_process_group("gloo")


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)


m = SimpleModel().to(f"cuda:{local_rank}")
try:
    ddp_m = DDP(m, device_ids=[local_rank], output_device=local_rank)
    x = torch.randn(4, 10, device=f"cuda:{local_rank}")
    y = ddp_m(x).sum()
    y.backward()
    print(f"Rank {local_rank}: Backward pass and gradient sync SUCCESS!")
except Exception as e:
    print(f"Rank {local_rank}: DDP error: {e}")

dist.destroy_process_group()

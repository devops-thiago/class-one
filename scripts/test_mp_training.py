import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def run_worker(rank, world_size):
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29535"
    dist.init_process_group(
        backend="gloo",
        init_method="tcp://127.0.0.1:29535",
        world_size=world_size,
        rank=rank,
    )

    # Simple model with multiple layers
    model = torch.nn.Sequential(
        torch.nn.Linear(64, 128), torch.nn.ReLU(), torch.nn.Linear(128, 64), torch.nn.ReLU(), torch.nn.Linear(64, 1)
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Run 25 steps to test socket stability with flat reduction
    for step in range(25):
        optimizer.zero_grad()
        x = torch.randn(8, 64, device=device)
        loss = model(x).sum()
        loss.backward()

        # Single flat buffer all_reduce
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        flat_grad = torch.cat([g.view(-1) for g in grads]).cpu()
        dist.all_reduce(flat_grad, op=dist.ReduceOp.SUM)
        flat_grad = flat_grad / world_size

        offset = 0
        for p in model.parameters():
            if p.grad is not None:
                numel = p.grad.numel()
                p.grad.data.copy_(flat_grad[offset : offset + numel].view_as(p.grad).to(device))
                offset += numel

        optimizer.step()

    print(f"Rank {rank}: 25 steps of flat reduction completed cleanly!")
    dist.destroy_process_group()


if __name__ == "__main__":
    world_size = 2
    mp.spawn(run_worker, args=(world_size,), nprocs=world_size, join=True)
    print("Flat buffer multi-GPU reduction test PASSED!")

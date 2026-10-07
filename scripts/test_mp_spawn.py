import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def worker(rank, world_size):
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    # TCP initialization on localhost
    dist.init_process_group(
        backend="gloo",
        init_method="tcp://127.0.0.1:29500",
        world_size=world_size,
        rank=rank,
    )

    # Test CPU all_reduce for parameter sync
    t = torch.tensor([float(rank + 1)])
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    print(f"Rank {rank} synchronized sum: {t.item()} (Expected: 3.0)")

    dist.destroy_process_group()


if __name__ == "__main__":
    world_size = 2
    mp.spawn(worker, args=(world_size,), nprocs=world_size, join=True)
    print("Multi-GPU distributed synchronization SUCCESS!")

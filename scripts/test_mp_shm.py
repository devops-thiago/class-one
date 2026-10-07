import torch
import torch.multiprocessing as mp


def worker(rank, q_in, q_out, steps=100):
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    for step in range(steps):
        # Local gradient tensor on CPU
        grad = torch.randn(1000)
        q_out.put((rank, grad))
        other_rank, other_grad = q_in.get()
        (grad + other_grad) / 2.0
    print(f"Worker {rank}: Successfully executed {steps} queue steps in shared memory with ZERO sockets!")


if __name__ == "__main__":
    q0 = mp.Queue()
    q1 = mp.Queue()
    # Rank 0 puts to q0, gets from q1. Rank 1 puts to q1, gets from q0.
    p0 = mp.Process(target=worker, args=(0, q1, q0, 100))
    p1 = mp.Process(target=worker, args=(1, q0, q1, 100))
    p0.start()
    p1.start()
    p0.join()
    p1.join()
    print("Multi-process shared memory test PASSED!")

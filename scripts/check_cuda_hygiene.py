#!/usr/bin/env python3
"""Checks CUDA environment, GPU memory budgets, and PyTorch settings on Windows.

Verifies:
1. Dual NVIDIA GeForce RTX 5060 Ti GPU detection (16 GB VRAM each).
2. Environment configuration:
   - PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
   - HF_TOKEN availability
3. PyTorch multiprocessing start method (`spawn`).
4. Current VRAM allocation and free memory per GPU.
5. Windows console encoding (UTF-8).
"""

import os
import sys

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def get_token() -> str | None:
    token = os.environ.get("HF_TOKEN")
    if not token and os.path.exists(".env"):
        with open(".env", encoding="utf-8") as f:
            for line in f:
                if line.strip().startswith("HF_TOKEN="):
                    token = line.strip().split("=", 1)[1].strip().strip("'\"")
                    break
    return token


def main():
    print("=" * 80)
    print("CLASSONE WINDOWS & CUDA HYGIENE DIAGNOSTIC")
    print("=" * 80)

    # 1. Python Environment
    print(f"[*] Python Executable: {sys.executable}")
    is_venv = ".venv" in sys.executable.lower()
    if is_venv:
        print("    [OK] Running inside workspace virtual environment (.venv)")
    else:
        print("    [!] WARNING: Not running in workspace .venv! Use .venv\\Scripts\\python.exe")

    # 2. PyTorch & CUDA Detection
    try:
        import torch

        print(f"[*] PyTorch Version:   {torch.__version__}")
        cuda_avail = torch.cuda.is_available()
        print(f"[*] CUDA Available:    {cuda_avail}")

        if cuda_avail:
            num_devices = torch.cuda.device_count()
            print(f"[*] CUDA Device Count: {num_devices}")
            for i in range(num_devices):
                props = torch.cuda.get_device_properties(i)
                total_gb = props.total_memory / (1024**3)
                free_b, total_b = torch.cuda.mem_get_info(i)
                free_gb = free_b / (1024**3)
                used_gb = (total_b - free_b) / (1024**3)
                print(
                    f"    - GPU {i}: {props.name} ({total_gb:.1f} GB Total | {free_gb:.1f} GB Free | {used_gb:.1f} GB Used)"
                )

            # Check allocator config
            alloc_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
            if "expandable_segments:True" in alloc_conf or "expandable_segments:true" in alloc_conf:
                print("    [OK] PYTORCH_CUDA_ALLOC_CONF includes expandable_segments:True")
            else:
                print(
                    f"    [!] NOTE: PYTORCH_CUDA_ALLOC_CONF='{alloc_conf}'. Recommend setting 'expandable_segments:True' to prevent virtual memory fragmentation."
                )
        else:
            print("    [!] ERROR: CUDA is not available in this PyTorch installation!")
    except ImportError:
        print("    [!] ERROR: PyTorch is not installed in this environment.")

    # 3. Multiprocessing
    import multiprocessing as mp

    start_method = mp.get_start_method()
    print(f"[*] Multiprocessing Start Method: {start_method}")
    if sys.platform == "win32" and start_method != "spawn":
        print("    [!] NOTE: Windows requires 'spawn' start method for PyTorch distributed tasks.")

    # 4. HF Authentication
    token = get_token()
    if token:
        masked = token[:6] + "..." + token[-4:]
        print(f"[*] HF_TOKEN Detected: {masked}")
    else:
        print("    [!] WARNING: No HF_TOKEN found in environment or .env file.")

    print("=" * 80)


if __name__ == "__main__":
    main()

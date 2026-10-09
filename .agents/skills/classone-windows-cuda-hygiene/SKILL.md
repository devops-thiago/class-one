---
name: classone-windows-cuda-hygiene
description: >-
  Manage Windows PowerShell execution, environment variables, PyTorch CUDA memory,
  and dual-GPU (2x RTX 5060 Ti 16GB) hygiene for ClassOne. Prevents PowerShell quote mangling,
  avoids WinSock 0xC0000409 stack buffer overruns, prevents CUDA OOM via non-reentrant gradient checkpointing
  and expandable segments, and enforces clean sequential evaluation workflows.
  Use when running multi-GPU training on Windows, configuring CUDA memory, running PowerShell commands,
  debugging OOM issues, or troubleshooting environment setup.
---

# ClassOne Windows & Dual-GPU CUDA Hygiene Runbook

This skill details mandatory patterns and solutions for executing Python commands, managing PyTorch GPU memory on 2x NVIDIA GeForce RTX 5060 Ti (16 GB each), and preventing common Windows runtime gotchas.

---

## 1. PowerShell Execution & Quoting Golden Rules

### The Problem
PowerShell parser strips outer quotation marks, mangles nested JSON quotes, and misinterprets parentheses and semicolons when executing inline commands (`python -c "..."`), causing cryptic syntax errors such as:
`ParserError: Unexpected token ')' in expression or statement.`

### Mandatory Rules
1. **Never use multi-line inline Python in PowerShell**:
   Always execute a standalone `.py` script or write a temporary script in `scratch/`.
2. **Always use the workspace virtual environment**:
   Execute with `.venv\Scripts\python.exe` instead of bare `python`. Bare `python` points to Windows App Store Python without PyTorch installed.
3. **Line Continuations**:
   In PowerShell, use the backtick (`` ` ``) for multi-line commands, NOT the bash backslash (`\`).
4. **UTF-8 Output Encoding**:
   Windows console defaults to `cp1252`, causing `UnicodeEncodeError: 'charmap' codec can't encode character '\u2713'`.
   Always set or include at the start of Python CLI scripts:
   ```python
   import sys
   if sys.platform == "win32":
       sys.stdout.reconfigure(encoding="utf-8")
       sys.stderr.reconfigure(encoding="utf-8")
   ```

---

## 2. Multi-GPU Training on Windows: WinSock Overrun Prevention

### The Problem
Using `torchrun` with Gloo backend over TCP sockets across separate processes on Windows triggers rapid socket calls that lead to:
`0xC0000409 (STATUS_STACK_BUFFER_OVERRUN) in WinSock (ws2_32.dll)`.

### The Proven Pattern
- **Do NOT use `torchrun` with Gloo sockets across separate OS processes on Windows.**
- **Use `torch.multiprocessing.Process` with shared-memory queues (`sync_gradients_shm`)**:
  - Flatten gradients into a single contiguous tensor.
  - Zero-fill parameters where `p.grad is None` (to ensure identical tensor lengths regardless of question distribution in the batch).
  - Cast gradient slices to `p.dtype` (BFloat16) when assigning.

---

## 3. CUDA Memory Budget & OOM Prevention (Dual RTX 5060 Ti 16 GB)

### Memory Budget Guidelines (Per GPU)
| Component | Budget (BF16) | Notes |
|---|---|---|
| **Base Model (4B/E4B)** | ~8.60 GB | BF16 base weights |
| **Trainable LoRA & Heads** | ~0.15 GB | $r=16, \alpha=32$ (~35M params) |
| **AdamW Optimizer States** | ~0.42 GB | First & second moments |
| **Activations (seq len 2048)** | ~1.20 GB | With gradient checkpointing enabled |
| **Total Allocated** | **~11.24 GB** | **Leaves ~4.76 GB headroom (30% safety margin)** |

### Strict OOM Hygiene Requirements
1. **Allocator Fragmentation**:
   Always set before PyTorch CUDA initialization:
   ```powershell
   $env:PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
   ```
   Or in Python before `import torch`:
   ```python
   os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
   ```
2. **Gradient Checkpointing**:
   Enable non-reentrant gradient checkpointing:
   ```python
   model.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
   ```
   This slashes activation memory by 80% (from ~5 GB to ~300 MB).
3. **Graph Reference Cleanup**:
   Explicitly delete intermediate activation subgraphs after the backward pass:
   ```python
   scaled_loss.backward()
   del item_loss, seq_hidden, hidden_states
   ```
4. **Zero-Grad Memory Release**:
   Call:
   ```python
   optimizer.zero_grad(set_to_none=True)
   ```
   Gradient memory is immediately freed rather than retained as zero-filled buffers.
5. **Periodic Cache Eviction**:
   Call `torch.cuda.empty_cache()` every 20–30 steps during training loops and after evaluating each model.

---

## 4. Environment & Diagnostic Quick-Check

To verify your environment, run the diagnostic tool:

```powershell
.venv\Scripts\python.exe scripts/check_cuda_hygiene.py
```

Expected Output:
- CUDA available with 2 devices detected.
- Running within `.venv`.
- `HF_TOKEN` loaded from `.env`.

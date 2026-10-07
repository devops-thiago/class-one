#!/usr/bin/env python3
"""Empirical test loading google/gemma-4-E4B-it into ClassOneModel on CUDA.
Measures base VRAM footprint, LoRA attachment, and executes a forward decision pass."""

import time

import torch
from transformers import AutoTokenizer

from classone.modeling.modeling_classone import ClassOneModel
from classone.schemas import ChoiceQuestion, NoulQuestion, ScoreQuestion
from classone.tokenizer import ClassOnePromptBuilder
from classone.train.trainer import ClassOneTrainer

MODEL_ID = "google/gemma-4-E4B-it"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def main():
    print("=" * 80)
    print(f"       EXERCISING {MODEL_ID} ON {DEVICE.upper()}")
    print("=" * 80)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        vram_start = torch.cuda.memory_allocated(0) / (1024**3)
        print(f"[*] Initial VRAM allocated: {vram_start:.2f} GB")

    # 1. Load Tokenizer
    print(f"\n[*] Loading tokenizer from {MODEL_ID}...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    builder = ClassOnePromptBuilder(tok)
    print(f"[✓] Tokenizer loaded (vocab size: {len(tok)})")

    # 2. Load Model into ClassOneModel
    print(f"\n[*] Loading backbone {MODEL_ID} (BF16)...")
    t0 = time.time()
    model = ClassOneModel.from_backbone(
        base_model_name_or_path=MODEL_ID,
        tokenizer=tok,
        device=DEVICE,
        torch_dtype=torch.bfloat16,
    )
    t_load = time.time() - t0
    print(f"[✓] Model loaded in {t_load:.1f}s!")

    if torch.cuda.is_available():
        vram_base = torch.cuda.memory_allocated(0) / (1024**3)
        vram_peak = torch.cuda.max_memory_allocated(0) / (1024**3)
        print(f"[*] Base Model VRAM allocated: {vram_base:.2f} GB (Peak during load: {vram_peak:.2f} GB)")

    # 3. Attach LoRA Adapters
    print("\n[*] Initializing ClassOneTrainer and attaching LoRA adapters (r=16, alpha=32)...")
    trainer = ClassOneTrainer(model=model, prompt_builder=builder, lr=2e-4, device=DEVICE)
    trainer.enable_lora(r=16, lora_alpha=32)

    if torch.cuda.is_available():
        vram_lora = torch.cuda.memory_allocated(0) / (1024**3)
        print(f"[✓] LoRA attached! Total VRAM with trainable adapters: {vram_lora:.2f} GB")

    # 4. Pack Multi-Primitive Decision Task
    print("\n[*] Packing sample decision sequence...")
    state = "Company T&E Policy: Lodging reimbursement is capped at $350/night under Endorsement L-2. Claim HX-9921 requests $420/night."
    questions = {
        "settlement": ChoiceQuestion(
            instructions="Determine how the lodging claim must be settled:",
            criteria={
                "pay_sublimit_350": "Payment capped at $350 endorsement sublimit",
                "pay_full_420": "Pay full lodging estimate without sublimit",
                "deny_unauthorized": "Deny reimbursement as unauthorized",
            },
        ),
        "is_capped": NoulQuestion(instructions="Does a policy endorsement sublimit cap the lodging reimbursement?"),
        "severity": ScoreQuestion(
            instructions="Rate policy compliance severity:", criteria=["compliant", "advisory_warning", "policy_breach"]
        ),
    }

    packed = builder.pack(state, questions)

    # 5. Execute Forward Pass
    print("[*] Executing forward evaluation pass...")
    model.eval()
    t0_eval = time.time()
    with torch.no_grad():
        answers = model.evaluate_packed(packed)
    t_eval = (time.time() - t0_eval) * 1000

    print(f"[✓] Forward pass executed in {t_eval:.1f} ms!")
    for q_id, ans in answers.items():
        if hasattr(ans, "choice"):
            print(
                f"  • {q_id:12s} (Choice) -> Winner: {ans.choice} | Conf: {ans.confidence:.4f} | Margin: {ans.margin:.4f}"
            )
        elif hasattr(ans, "noul"):
            print(f"  • {q_id:12s} (Noul)   -> Prob: {ans.noul:.4f}")
        elif hasattr(ans, "score"):
            print(f"  • {q_id:12s} (Score)  -> Score: {ans.score:.2f} | Conf: {ans.confidence:.4f}")

    if torch.cuda.is_available():
        vram_final = torch.cuda.memory_allocated(0) / (1024**3)
        vram_peak_final = torch.cuda.max_memory_allocated(0) / (1024**3)
        print(f"\n[*] Final VRAM usage: {vram_final:.2f} GB (Total Peak: {vram_peak_final:.2f} GB)")
        print(f"[✓] Headroom remaining on GPU 0: {16.0 - vram_peak_final:.2f} GB free!")

    print("\n" + "=" * 80)
    print("       GEMMA 4 E4B EMPIRICAL EXERCISE COMPLETE: SUCCESS")
    print("=" * 80)


if __name__ == "__main__":
    main()

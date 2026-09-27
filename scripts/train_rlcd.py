#!/usr/bin/env python3
"""CLI utility to train a ClassOne System 1 decision model using LoRA and RLCD loss.

Supports:
- Single-GPU (CUDA / Apple Silicon MPS / CPU)
- Multi-GPU DistributedDataParallel (DDP) via torchrun (e.g. 2x RTX 16GB GPUs)
"""

import argparse
import os
import sys

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer

from classone.data.dataset import (
    ClassOneDataset,
    bundle_multi_task_items,
    classone_collate_fn,
    create_classone_dataloader,
)
from classone.modeling.configuration_classone import ClassOneConfig
from classone.modeling.modeling_classone import ClassOneModel
from classone.schemas import ChoiceQuestion, NoulQuestion, ScoreQuestion
from classone.tokenizer import ClassOnePromptBuilder
from classone.train.trainer import ClassOneTrainer


def generate_synthetic_dataset(num_samples: int = 50) -> ClassOneDataset:
    """Generates a diverse synthetic multi-task dataset for fine-tuning & calibration."""
    items = []
    topics = [
        ("My monthly charge was doubled and I demand my money back.", "billing", 1.0, 3),
        ("The iOS app crashes every time I open the settings page.", "tech", 0.0, 2),
        ("Can we schedule an enterprise demo for our 200-person team?", "sales", 0.0, 1),
        ("Someone accessed my account from an unknown IP address.", "security", 0.0, 3),
        ("Cancel my subscription immediately, service is unusable.", "billing", 1.0, 3),
        ("Where can I find documentation on the API webhook signature?", "tech", 0.0, 1),
    ]

    all_criteria = {
        "billing": "Charges, invoices, and refund disputes",
        "tech": "Software bugs, application crashes, and API issues",
        "sales": "Enterprise upgrades, contracts, and sales demos",
        "security": "Unauthorized access, suspicious activity, and MFA",
    }
    rubric = ["mild", "moderate", "severe"]

    for i in range(num_samples):
        text, dept, is_refund, urgency = topics[i % len(topics)]
        state = {
            "ticket_id": 1000 + i,
            "message": f"[{i}] {text}",
        }
        item = bundle_multi_task_items(
            state=state,
            questions={
                "is_refund": NoulQuestion(instructions="Is customer asking for a refund or money back?"),
                "department": ChoiceQuestion(
                    instructions="Assign ticket to department queue:",
                    criteria=all_criteria,
                ),
                "severity": ScoreQuestion(
                    instructions="Assess ticket urgency:",
                    criteria=rubric,
                ),
            },
            targets={
                "is_refund": is_refund,
                "department": dept,
                "severity": urgency,
            },
        )
        items.append(item)

    return ClassOneDataset(items)


def parse_args():
    parser = argparse.ArgumentParser(description="Train ClassOne System 1 Decision Model via RLCD")
    parser.add_argument(
        "--model",
        type=str,
        default="google/gemma-4-e2b-it",
        help="Backbone model ID or path (e.g. google/gemma-4-e2b-it, google/gemma-2-2b-it, or standalone)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to JSONL dataset. If omitted, generates synthetic training samples.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=60,
        help="Number of synthetic samples to generate if --data is not provided",
    )
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size per GPU")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha scaling")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "mps", "cuda", "cpu"],
        help="Target training device",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./checkpoints/classone_v1",
        help="Directory to save fine-tuned heads and LoRA weights",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        default=True,
        help="Run post-hoc temperature calibration after training",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Detect multi-GPU distributed environment (torchrun)
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    is_distributed = local_rank != -1

    if is_distributed:
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
        backend = "gloo" if sys.platform == "win32" else "nccl"
        dist.init_process_group(backend=backend)
        global_rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        global_rank = 0
        world_size = 1
        if args.device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        else:
            device = args.device

    is_main_process = global_rank == 0

    if is_main_process:
        print(f"[*] Initializing ClassOne Training (World Size: {world_size}, Device: {device})")

    # Initialize Tokenizer and Model
    if args.model == "standalone":
        if is_main_process:
            print("[*] Using standalone lightweight transformer backbone (rapid test mode)")
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import PreTrainedTokenizerFast

        vocab = {"[UNK]": 0, "[PAD]": 1, "[CLS]": 2, "[SEP]": 3, "[MASK]": 4}
        tok = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
        tok.pre_tokenizer = Whitespace()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]", pad_token="[PAD]")
        builder = ClassOnePromptBuilder(tokenizer)
        config = ClassOneConfig(hidden_size=128, head_hidden_size=64)
        model = ClassOneModel(config)
    else:
        if is_main_process:
            print(f"[*] Loading pretrained HuggingFace backbone: {args.model}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.model)
            builder = ClassOnePromptBuilder(tokenizer)
            model = ClassOneModel.from_backbone(
                base_model_name_or_path=args.model,
                tokenizer=tokenizer,
                device=device,
                torch_dtype=torch.bfloat16
                if "cuda" in str(device)
                else (torch.float16 if device == "mps" else torch.float32),
            )
        except Exception as exc:
            if is_main_process:
                print(f"[!] Error loading backbone '{args.model}': {exc}")
                print("[!] If this is a gated model (like Gemma), ensure HF_TOKEN is exported.")
            sys.exit(1)

    # Initialize Trainer
    trainer = ClassOneTrainer(
        model=model,
        prompt_builder=builder,
        lr=args.lr,
        device=device,
    )

    # Attach LoRA if not standalone
    if args.model != "standalone":
        if is_main_process:
            print(f"[*] Attaching LoRA adapters (r={args.lora_r}, alpha={args.lora_alpha})...")
        trainer.enable_lora(r=args.lora_r, lora_alpha=args.lora_alpha)

    # Wrap with DistributedDataParallel if using multiple GPUs
    if is_distributed:
        trainer.model = DDP(trainer.model, device_ids=[local_rank], find_unused_parameters=True)

    # Load or generate dataset
    if args.data:
        if not os.path.exists(args.data):
            if is_main_process:
                print(f"[!] Specified dataset file does not exist: {args.data}")
            sys.exit(1)
        if is_main_process:
            print(f"[*] Loading training data from: {args.data}")
        dataset = ClassOneDataset.load_jsonl(args.data)
    else:
        if is_main_process:
            print(f"[*] Generating synthetic multi-task training dataset ({args.num_samples} samples)...")
        dataset = generate_synthetic_dataset(num_samples=args.num_samples)

    # Configure Distributed DataLoader
    if is_distributed:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=global_rank, shuffle=True)
        train_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            collate_fn=classone_collate_fn,
        )
    else:
        train_loader = create_classone_dataloader(dataset, batch_size=args.batch_size, shuffle=True)

    if is_main_process:
        print(f"[*] Total dataset size: {len(dataset)} items | Per-GPU Batch size: {args.batch_size}")
        print("\n=== Starting RLCD Multi-Task Fine-Tuning ===")

    # Training Loop
    for epoch in range(1, args.epochs + 1):
        if is_distributed:
            sampler.set_epoch(epoch)

        epoch_loss = 0.0
        total_q = 0
        step_count = 0

        for batch in train_loader:
            step_count += 1
            metrics = trainer.train_step(batch)
            epoch_loss += metrics["loss"]
            total_q += metrics["questions"]

            if is_main_process and (step_count % 5 == 0 or step_count == len(train_loader)):
                print(
                    f"  Epoch [{epoch}/{args.epochs}] | Step [{step_count}/{len(train_loader)}] | "
                    f"RLCD Loss: {metrics['loss']:.4f} ({metrics['questions']} questions)"
                )

        if is_main_process:
            avg_epoch_loss = epoch_loss / max(1, step_count)
            print(f"[*] Epoch {epoch} Completed | Average Loss: {avg_epoch_loss:.4f}\n")

    # Temperature Calibration & Checkpoint (Rank 0 only)
    if is_main_process:
        # Unwrap DDP if distributed
        raw_model = trainer.model.module if is_distributed else trainer.model
        eval_trainer = ClassOneTrainer(model=raw_model, prompt_builder=builder, device=device)

        if args.calibrate:
            print("[*] Running post-hoc temperature calibration on validation split...")
            val_samples = dataset.items[:10]
            calibrated_temps = eval_trainer.calibrate_temperature(val_samples, lr=0.05, max_epochs=15)
            print(f"  Calibrated Noul Temp:   {calibrated_temps['noul_temp']:.4f}")
            print(f"  Calibrated Choice Temp: {calibrated_temps['choice_temp']:.4f}")
            print(f"  Calibrated Score Temp:  {calibrated_temps['score_temp']:.4f}")

        print(f"\n[*] Saving trained model checkpoint to: {args.output_dir}")
        eval_trainer.save_checkpoint(args.output_dir)
        print("[OK] Training & checkpoint export completed successfully!")

    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

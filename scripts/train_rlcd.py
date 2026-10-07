#!/usr/bin/env python3
"""Multi-GPU Distributed Fine-Tuning & Calibration script for ClassOne System 1 Decision Model.

Trains ClassOne decision heads and PEFT LoRA adapters using RLCD (Proper Scoring + Contrastive Margin Loss)
across single or multiple GPUs (NVIDIA RTX 5060 Ti x 2).

Usage:
    # Train across both GPUs automatically:
    python scripts/train_rlcd.py --data data/training_corpus.jsonl --gpus 2 --epochs 3 --batch-size 4

    # Single GPU:
    python scripts/train_rlcd.py --data data/training_corpus.jsonl --gpus 1
"""

import argparse
import os
import sys
import time

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.multiprocessing as mp
from transformers import AutoTokenizer

from classone.data.dataset import (
    ClassOneDataset,
    bundle_multi_task_items,
    create_classone_dataloader,
)
from classone.modeling.configuration_classone import ClassOneConfig
from classone.modeling.modeling_classone import ClassOneModel
from classone.schemas import ChoiceQuestion, NoulQuestion, ScoreQuestion
from classone.tokenizer import ClassOnePromptBuilder
from classone.train.trainer import ClassOneTrainer


def generate_synthetic_dataset(num_samples: int = 50) -> ClassOneDataset:
    """Generates a synthetic multi-task dataset if no data file is provided."""
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
                "dept": ChoiceQuestion(
                    instructions="Route this support ticket to the appropriate department:",
                    criteria=all_criteria,
                ),
                "refund_request": NoulQuestion(
                    instructions="Does the customer explicitly request a refund or billing adjustment?",
                ),
                "urgency": ScoreQuestion(
                    instructions="Rate the operational urgency of this ticket:",
                    criteria=rubric,
                ),
            },
            targets={
                "dept": dept,
                "refund_request": is_refund,
                "urgency": urgency,
            },
        )
        items.append(item)

    return ClassOneDataset(items)


def parse_args():
    parser = argparse.ArgumentParser(description="Train ClassOne System 1 Decision Model via RLCD")
    parser.add_argument(
        "--model",
        type=str,
        default="devops-thiago/classone-gemma4-e2b",
        help="Backbone model ID or path (default: devops-thiago/classone-gemma4-e2b)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/training_corpus.jsonl",
        help="Path to JSONL dataset. If omitted, generates synthetic training samples.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=60,
        help="Number of synthetic samples to generate if --data is not provided",
    )
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size per GPU")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha scaling")
    parser.add_argument(
        "--gpus",
        type=int,
        default=None,
        help="Number of GPUs to use for training (auto-detects available devices)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints/classone_gemma4_e2b",
        help="Directory to save fine-tuned heads and LoRA weights",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        default=True,
        help="Run post-hoc temperature calibration after training",
    )
    parser.add_argument(
        "--pretrained-heads",
        type=str,
        default=None,
        help="Path to pre-trained classone_heads.pt file to initialize decision heads",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default=None,
        choices=["4bit", "8bit", "nf4"],
        help="Quantization format for base model weights ('4bit' or '8bit')",
    )
    return parser.parse_args()


def run_training_worker(rank: int, world_size: int, args, q_out=None, q_in=None):
    """Worker function executed per GPU."""
    is_main_process = rank == 0

    if world_size > 1:
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if is_main_process:
        print(f"[*] Initializing ClassOne Training on {world_size} GPU(s)...")

    # Initialize Tokenizer and Model
    if args.model == "standalone":
        if is_main_process:
            print("[*] Using standalone lightweight transformer backbone")
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
            print(f"[*] Loading backbone: {args.model}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.model)
            builder = ClassOnePromptBuilder(tokenizer)
            model = ClassOneModel.from_backbone(
                base_model_name_or_path=args.model,
                tokenizer=tokenizer,
                device=device,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                quantization=args.quantization,
            )
        except Exception as exc:
            if is_main_process:
                print(f"[!] Error loading backbone '{args.model}': {exc}")
            sys.exit(1)

    # Initialize Trainer and LoRA
    trainer = ClassOneTrainer(
        model=model,
        prompt_builder=builder,
        lr=args.lr,
        device=device,
    )

    # Load pre-trained decision heads if fine-tuning from an existing ClassOne checkpoint or local heads
    if args.pretrained_heads and os.path.exists(args.pretrained_heads):
        if is_main_process:
            print(f"[*] Loading pre-trained decision heads from local file {args.pretrained_heads}...")
        try:
            heads_data = torch.load(args.pretrained_heads, map_location=device)
            model.noul_head.load_state_dict(heads_data["noul_head"], strict=False)
            model.choice_head.load_state_dict(heads_data["choice_head"], strict=False)
            model.score_head.load_state_dict(heads_data["score_head"], strict=False)
            if is_main_process:
                print("[✓] Local pre-trained heads loaded successfully.")
        except Exception as e:
            if is_main_process:
                print(f"[!] Warning: Failed to load local pre-trained heads: {e}")
    elif args.model == "devops-thiago/classone-gemma4-e2b":
        if is_main_process:
            print(f"[*] Loading pre-trained decision heads from {args.model}...")
        from huggingface_hub import hf_hub_download

        try:
            heads_path = hf_hub_download(args.model, "classone_heads.pt")
            heads_data = torch.load(heads_path, map_location=device)
            model.noul_head.load_state_dict(heads_data["noul_head"], strict=False)
            model.choice_head.load_state_dict(heads_data["choice_head"], strict=False)
            model.score_head.load_state_dict(heads_data["score_head"], strict=False)
            if is_main_process:
                print("[✓] Pre-trained heads loaded successfully.")
        except Exception as e:
            if is_main_process:
                print(f"[!] Warning: Failed to load pre-trained heads: {e}")

    if args.model != "standalone":
        if is_main_process:
            print(f"[*] Attaching LoRA adapters (r={args.lora_r}, alpha={args.lora_alpha})...")
        trainer.enable_lora(r=args.lora_r, lora_alpha=args.lora_alpha)
        if hasattr(model.backbone, "gradient_checkpointing_enable"):
            model.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            if is_main_process:
                print("[*] Gradient checkpointing enabled (80% activation memory reduction)")

    # Load dataset
    if args.data and os.path.exists(args.data):
        full_dataset = ClassOneDataset.load_jsonl(args.data)
        if is_main_process:
            print(f"[*] Loaded {len(full_dataset)} items from {args.data}")
    else:
        full_dataset = generate_synthetic_dataset(num_samples=args.num_samples)
        if is_main_process:
            print(f"[*] Generated {len(full_dataset)} synthetic items")

    # Shard dataset across GPUs
    if world_size > 1:
        rank_items = [full_dataset.items[i] for i in range(len(full_dataset.items)) if i % world_size == rank]
        local_dataset = ClassOneDataset(rank_items)
    else:
        local_dataset = full_dataset

    train_loader = create_classone_dataloader(local_dataset, batch_size=args.batch_size, shuffle=True)

    if is_main_process:
        print(f"[*] Total dataset size: {len(full_dataset)} items | Shard per GPU: {len(local_dataset)} items")
        print("\n=== Starting RLCD Multi-GPU Fine-Tuning ===")

    # Training Loop
    total_steps = len(train_loader) * args.epochs
    trainer.setup_scheduler(total_steps=total_steps, warmup_steps=min(200, max(50, total_steps // 15)))

    t0_train = time.time()
    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        step_count = 0

        for batch in train_loader:
            step_count += 1
            metrics = trainer.train_step(batch, world_size=world_size, q_out=q_out, q_in=q_in)
            epoch_loss += metrics["loss"]

            if step_count % 20 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

            if is_main_process and (step_count % 25 == 0 or step_count == len(train_loader)):
                cur_lr = trainer.optimizer.param_groups[0]["lr"]
                print(
                    f"  Epoch [{epoch}/{args.epochs}] | Step [{step_count}/{len(train_loader)}] | "
                    f"LR: {cur_lr:.2e} | RLCD Loss: {metrics['loss']:.4f}"
                )

        if is_main_process:
            avg_epoch_loss = epoch_loss / max(1, step_count)
            print(f"[*] Epoch {epoch} Completed | Average Loss: {avg_epoch_loss:.4f}\n")

    train_time = time.time() - t0_train
    if is_main_process:
        print(f"[✓] Multi-GPU Training finished in {train_time:.1f}s")

    # Post-hoc Calibration & Checkpoint Export (Rank 0 only)
    if is_main_process:
        if args.calibrate:
            print("[*] Running post-hoc temperature calibration...")
            val_samples = full_dataset.items[:25]
            calibrated_temps = trainer.calibrate_temperature(val_samples, lr=0.05, max_epochs=15)
            print(f"  Calibrated Noul Temp:   {calibrated_temps['noul_temp']:.4f}")
            print(f"  Calibrated Choice Temp: {calibrated_temps['choice_temp']:.4f}")
            print(f"  Calibrated Score Temp:  {calibrated_temps['score_temp']:.4f}")

        print(f"[*] Saving model checkpoint to: {args.output_dir}")
        trainer.save_checkpoint(args.output_dir)
        print("[✓] Model checkpoint saved successfully!")


def main():
    args = parse_args()

    num_gpus = args.gpus
    if num_gpus is None:
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1

    if num_gpus > 1:
        print(f"[*] Launching Multi-GPU training on {num_gpus} devices using Shared-Memory IPC...")
        q0 = mp.Queue(maxsize=2)
        q1 = mp.Queue(maxsize=2)
        p0 = mp.Process(target=run_training_worker, args=(0, num_gpus, args, q0, q1))
        p1 = mp.Process(target=run_training_worker, args=(1, num_gpus, args, q1, q0))
        p0.start()
        p1.start()
        p0.join()
        p1.join()
        if p0.exitcode != 0 or p1.exitcode != 0:
            sys.exit(max(p0.exitcode or 0, p1.exitcode or 0))
    else:
        run_training_worker(rank=0, world_size=1, args=args)


if __name__ == "__main__":
    main()

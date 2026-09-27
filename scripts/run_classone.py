#!/usr/bin/env python3
"""CLI utility to run Jev System One inference on Gemma 4 E2B or any local/HF backbone."""

import argparse
import json
import sys

import torch
from transformers import AutoTokenizer

from classone.modeling.modeling_classone import ClassOneModel
from classone.schemas import ChoiceQuestion, NoulQuestion, ScoreQuestion
from classone.tokenizer import ClassOnePromptBuilder


def parse_args():
    parser = argparse.ArgumentParser(description="Run single-pass Jev decision inference using Gemma backbone.")
    parser.add_argument(
        "--model",
        type=str,
        default="google/gemma-4-e2b-it",
        help="HuggingFace model ID or local directory (e.g. google/gemma-4-e2b-it, google/gemma-2-2b-it)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to trained checkpoint directory (e.g. ./checkpoints/classone_gemma4_e2b)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "mps", "cuda", "cpu"],
        help="Target hardware device (default: auto)",
    )
    parser.add_argument(
        "--state",
        type=str,
        default='{"customer": "Alex", "query": "I would like to cancel my subscription and get a refund."}',
        help="Input state (JSON string or raw text)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Determine device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    print(f"[*] Initializing Jev on device: {device}")
    print(f"[*] Loading backbone: {args.model}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        builder = ClassOnePromptBuilder(tokenizer)
        model = ClassOneModel.from_backbone(
            base_model_name_or_path=args.model,
            tokenizer=tokenizer,
            device=device,
            torch_dtype=torch.float16 if device in ["cuda", "mps"] else torch.float32,
        )
    except Exception as e:
        print("[!] Note: If this is a gated model, make sure HF_TOKEN is exported.")
        print(f"[!] Error loading backbone: {e}")
        sys.exit(1)

    if args.checkpoint:
        print(f"[*] Loading fine-tuned weights and LoRA adapters from: {args.checkpoint}")
        from classone.train.trainer import ClassOneTrainer

        trainer = ClassOneTrainer(model=model, prompt_builder=builder, device=device)
        trainer.load_checkpoint(args.checkpoint)

    # Parse state
    try:
        state = json.loads(args.state)
    except Exception:
        state = args.state

    # Define standard decision questions
    questions = {
        "refund_request": NoulQuestion(instructions="Is the user requesting a refund or money back?"),
        "department": ChoiceQuestion(
            instructions="Which support queue should handle this ticket?",
            criteria={
                "billing": "Invoices, payment charges, and refund requests",
                "technical": "Software bugs, error messages, and outages",
                "sales": "Plan upgrades and enterprise licensing",
            },
        ),
        "sentiment": ScoreQuestion(
            instructions="Assess customer dissatisfaction level:",
            criteria=["satisfied", "neutral", "dissatisfied", "churning"],
        ),
    }

    print("\n[*] Input State:")
    print(json.dumps(state, indent=2) if isinstance(state, dict) else state)

    packed = builder.pack(state=state, questions=questions)
    print(f"[*] Packed sequence length: {packed.input_ids.shape[1]} tokens")

    print("[*] Running parallel single-pass evaluation...")
    results = model.evaluate_packed(packed)

    print("\n=== Jev Decision Outputs ===")
    for q_id, res in results.items():
        print(f"\nQuestion: {q_id} ({res.type})")
        if res.type == "noul":
            print(f"  P(true): {res.noul:.4f}")
        elif res.type == "choice":
            print(f"  Selected: {res.choice} (Confidence: {res.confidence:.4f})")
            print(f"  Distribution: {res.probabilities}")
        elif res.type == "score":
            print(f"  Score: {res.score:.4f} (Confidence: {res.confidence:.4f})")
            print(f"  Distribution: {res.probabilities}")


if __name__ == "__main__":
    main()

import torch
import torch.multiprocessing as mp
from transformers import AutoTokenizer

from classone.modeling.modeling_classone import ClassOneModel
from classone.schemas import ChoiceQuestion, NoulQuestion
from classone.tokenizer import ClassOnePromptBuilder
from classone.train.trainer import ClassOneTrainer, TrainingItem


def run_worker(rank, world_size, q_out, q_in, steps=5):
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    print(f"Rank {rank} starting on {device}...")

    model_id = "devops-thiago/classone-gemma4-e2b"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    builder = ClassOnePromptBuilder(tokenizer)

    model = ClassOneModel.from_backbone(
        base_model_name_or_path=model_id,
        tokenizer=tokenizer,
        device=device,
        torch_dtype=torch.bfloat16,
    )
    trainer = ClassOneTrainer(model=model, prompt_builder=builder, lr=2e-4, device=device)
    trainer.enable_lora(r=16, lora_alpha=32)

    item = TrainingItem(
        state={"message": f"Server {rank} reported memory limit reached."},
        questions={
            "is_alert": NoulQuestion(instructions="Is this an alert?"),
            "team": ChoiceQuestion(instructions="Routing:", criteria={"infra": "Infra", "app": "App"}),
        },
        targets={"is_alert": 1.0, "team": "infra"},
    )

    for s in range(steps):
        # train step
        trainer.model.train()
        trainer.optimizer.zero_grad()
        packed = builder.pack(item.state, item.questions)
        hidden = model.extract_hidden_states(packed.input_ids.to(device), packed.attention_mask.to(device))
        seq_h = hidden[0].to(next(model.noul_head.parameters()).dtype)
        prob, logit = model.noul_head(seq_h[packed.questions["is_alert"].query_token_idx], return_logits=True)
        loss = trainer.loss_fn.forward_binary(prob, torch.tensor([1.0], device=device), logit=logit)
        loss.backward()

        # Shared memory sync
        grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
        flat = torch.cat([g.view(-1) for g in grads]).cpu()
        q_out.put(flat)
        other = q_in.get()
        avg = (flat + other) / 2.0

        offset = 0
        for p in model.parameters():
            if p.requires_grad and p.grad is not None:
                numel = p.grad.numel()
                p.grad.data.copy_(avg[offset : offset + numel].view_as(p.grad).to(device))
                offset += numel
        trainer.optimizer.step()
        print(f"Rank {rank} Step {s + 1}/{steps} Loss: {loss.item():.4f}")


if __name__ == "__main__":
    world_size = 2
    q0 = mp.Queue()
    q1 = mp.Queue()
    p0 = mp.Process(target=run_worker, args=(0, world_size, q0, q1, 5))
    p1 = mp.Process(target=run_worker, args=(1, world_size, q1, q0, 5))
    p0.start()
    p1.start()
    p0.join()
    p1.join()
    print("Multi-GPU training via shared memory Queue SUCCESSFUL!")

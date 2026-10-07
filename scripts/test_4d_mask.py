import torch
from transformers import AutoTokenizer

from classone.modeling.modeling_classone import ClassOneModel
from classone.schemas import ChoiceQuestion
from classone.tokenizer import ClassOnePromptBuilder

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model_id = "devops-thiago/classone-gemma4-e2b"
tok = AutoTokenizer.from_pretrained(model_id)
builder = ClassOnePromptBuilder(tok)
model = ClassOneModel.from_backbone(model_id, tokenizer=tok, device=device, torch_dtype=torch.bfloat16)

# Create a packed sequence with 3 options
q = ChoiceQuestion(
    instructions="Select intent:",
    criteria={"A": "Option Alpha text", "B": "Option Beta text", "C": "Option Gamma text"},
)
packed = builder.pack(state={"message": "Customer wants to cancel."}, questions={"q": q})
input_ids = packed.input_ids.to(device)
seq_len = input_ids.shape[1]

# Build custom 4D position-invariant mask
causal_2d = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=device))
mask_4d = torch.where(causal_2d, 0.0, float("-1e9")).to(dtype=torch.bfloat16)[None, None, :, :]
pos_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0)

print(f"Testing forward pass with 4D mask {mask_4d.shape} and position_ids {pos_ids.shape}...")
with torch.no_grad():
    outputs = model.backbone(input_ids=input_ids, attention_mask=mask_4d, position_ids=pos_ids)
    hidden = outputs[0] if isinstance(outputs, tuple) else outputs.last_hidden_state
    print(f"[✓] Backbone forward pass successful! Output shape: {hidden.shape}")

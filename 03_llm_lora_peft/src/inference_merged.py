from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

import torch


MODEL_PATH = "./checkpoints/merged_model"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.float32,
)

model.eval()

prompt = """
### Instruction:
猫とは？

### Response:
"""

inputs = tokenizer(
    prompt,
    return_tensors="pt",
)

with torch.no_grad():

    outputs = model.generate(
        **inputs,
        max_new_tokens=64,
        do_sample=False,
    )

response = tokenizer.decode(
    outputs[0],
    skip_special_tokens=True,
)

print(response)
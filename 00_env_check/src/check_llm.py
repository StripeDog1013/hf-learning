import torch
from transformers import pipeline

print("=== LLM Inference Check ===")

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

if torch.cuda.is_available():
    device = 0
    print("Using device: CUDA")
elif torch.backends.mps.is_available():
    device = "mps"
    print("Using device: MPS")
else:
    device = -1
    print("Using device: CPU")

pipe = pipeline(
    "text-generation",
    model=MODEL_NAME,
    device=device,
)

prompt = "こんにちは。あなたは何ができますか？"

result = pipe(
    prompt,
    max_new_tokens=100,
    do_sample=True,
    temperature=0.7,
)

print("\n--- Prompt ---")
print(prompt)

print("\n--- Output ---")
print(result[0]["generated_text"])
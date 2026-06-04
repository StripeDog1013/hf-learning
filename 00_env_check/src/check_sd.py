import torch
from diffusers import DiffusionPipeline

print("=== Stable Diffusion Check ===")

MODEL_NAME = "runwayml/stable-diffusion-v1-5"

if torch.cuda.is_available():
    device = "cuda"
    dtype = torch.float16
    print("Using device: CUDA")
elif torch.backends.mps.is_available():
    device = "mps"
    dtype = torch.float16
    print("Using device: MPS")
else:
    device = "cpu"
    dtype = torch.float32
    print("Using device: CPU")

pipe = DiffusionPipeline.from_pretrained(
    MODEL_NAME,
    torch_dtype=dtype,
)

pipe = pipe.to(device)

prompt = "a cute cream-colored cat with blue eyes, sitting on a wooden chair"

image = pipe(
    prompt,
    num_inference_steps=20,
).images[0]

output_path = "sd_test.png"
image.save(output_path)

print(f"Image saved to: {output_path}")
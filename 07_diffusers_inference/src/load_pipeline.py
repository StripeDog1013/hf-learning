import os

from config import (
    USE_CUDA_VISIBLE_DEVICES,
    PHYSICAL_CUDA_ID,
)

if USE_CUDA_VISIBLE_DEVICES:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(PHYSICAL_CUDA_ID)

from diffusers import StableDiffusionPipeline

from config import MODEL_NAME, CUDA_ID
from device import get_device, get_torch_dtype


def load_text_to_image_pipeline():
    device = get_device(cuda_id=CUDA_ID)
    torch_dtype = get_torch_dtype(device)

    print(f"Loading pipeline: {MODEL_NAME}")
    print(f"Device: {device}")
    print(f"dtype : {torch_dtype}")

    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch_dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )

    pipe = pipe.to(device)

    if device.startswith("cuda"):
        pipe.enable_attention_slicing()

    return pipe


def main():
    pipe = load_text_to_image_pipeline()
    print("===============================")
    print("Pipeline loaded successfully.")
    print(f"type:{type(pipe)}")


if __name__ == "__main__":
    main()
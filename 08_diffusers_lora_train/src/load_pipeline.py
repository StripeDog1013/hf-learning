import os

from config import (
    USE_CUDA_VISIBLE_DEVICES,
    PHYSICAL_CUDA_ID,
)

if USE_CUDA_VISIBLE_DEVICES:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(PHYSICAL_CUDA_ID)

import torch

from diffusers import StableDiffusionPipeline

from config import (
    MODEL_NAME,
    CUDA_ID,
    LORA_OUTPUT_DIR,
)

from device import (
    get_device,
    get_torch_dtype,
)


def load_base_pipeline():
    device = get_device(cuda_id=CUDA_ID)
    torch_dtype = get_torch_dtype(device)

    print(f"Loading base pipeline: {MODEL_NAME}")
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


def load_lora_pipeline(
    lora_path=LORA_OUTPUT_DIR,
    lora_scale=1.0,
):
    pipe = load_base_pipeline()

    print(f"Loading LoRA weights: {lora_path}")

    pipe.load_lora_weights(
        lora_path,
    )

    pipe.fuse_lora(
        lora_scale=lora_scale,
    )

    return pipe


def freeze_parameters(module):
    for param in module.parameters():
        param.requires_grad = False


def prepare_train_components():
    device = get_device(cuda_id=CUDA_ID)
    torch_dtype = get_torch_dtype(device)

    print(f"Loading training components: {MODEL_NAME}")
    print(f"Device: {device}")
    print(f"dtype : {torch_dtype}")

    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch_dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )

    pipe = pipe.to(device)

    freeze_parameters(pipe.vae)
    freeze_parameters(pipe.text_encoder)

    pipe.unet.train()

    return {
        "pipe": pipe,
        "vae": pipe.vae,
        "text_encoder": pipe.text_encoder,
        "tokenizer": pipe.tokenizer,
        "unet": pipe.unet,
        "noise_scheduler": pipe.scheduler,
        "device": device,
        "torch_dtype": torch_dtype,
    }


def main():
    pipe = load_base_pipeline()

    print("\nPipeline loaded successfully.")
    print(type(pipe))

    components = prepare_train_components()

    print("\nTraining components loaded.")
    print(f"UNet Training Mode: {components['unet'].training}")


if __name__ == "__main__":
    main()
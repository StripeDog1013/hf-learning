import os

from config import (
    BASE_MODEL_NAME,
    CUDA_ID,
    GUIDANCE_SCALE,
    HEIGHT,
    NEGATIVE_PROMPT,
    NUM_INFERENCE_STEPS,
    OUTPUT_DIR,
    PHYSICAL_CUDA_ID,
    SEED,
    USE_CUDA_VISIBLE_DEVICES,
    WIDTH,
    create_dirs,
)

if USE_CUDA_VISIBLE_DEVICES:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(
        PHYSICAL_CUDA_ID
    )

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from diffusers import StableDiffusionXLPipeline

from device import (
    clear_cuda_cache,
    get_torch_device,
    print_device_info,
)
from image_utils import save_image
from utils import (
    get_timestamp,
    print_header,
    save_generation_log,
    save_image_prompt,
    set_seed,
)


PROMPTS = [
    "A cute Shiba Inu wearing a blue jacket, realistic, high quality",
    "A futuristic city at night, neon lights, cinematic, high detail",
    "A small robot reading a book in a cozy room, warm lighting",
    "A fantasy castle floating above clouds, epic, detailed",
    "A bowl of ramen on a wooden table, realistic food photography",
]


def load_pipeline(
    device: torch.device,
):
    dtype = (
        torch.float16
        if device.type == "cuda"
        else torch.float32
    )

    pipe = StableDiffusionXLPipeline.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=dtype,
        variant=(
            "fp16"
            if device.type == "cuda"
            else None
        ),
        use_safetensors=True,
    )

    pipe.to(device)

    try:
        pipe.enable_xformers_memory_efficient_attention()
        print_header("xFormers enabled.")
    except Exception as e:
        print_header(f"xFormers not enabled: {e}")

    pipe.text_encoder.to(device)
    pipe.text_encoder_2.to(device)
    pipe.unet.to(device)
    pipe.vae.to(device)

    return pipe


def generate_image(
    pipe,
    prompt: str,
    seed: int,
    device: torch.device,
):
    generator = torch.Generator(
        device=device
    ).manual_seed(seed)

    image = pipe(
        prompt=prompt,
        negative_prompt=NEGATIVE_PROMPT,
        height=HEIGHT,
        width=WIDTH,
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        generator=generator,
    ).images[0]

    return image


def main():

    print_header("Batch Generate SDXL")

    create_dirs()
    set_seed(SEED)

    device = get_torch_device(
        cuda_id=CUDA_ID
    )

    print_device_info(
        cuda_id=CUDA_ID
    )

    pipe = load_pipeline(
        device=device
    )

    logs = []

    batch_timestamp = get_timestamp()

    for i, prompt in enumerate(
        PROMPTS,
        start=1,
    ):

        seed = SEED + i

        print_header(
            f"Generate {i}/{len(PROMPTS)}"
        )

        print(prompt)

        image = generate_image(
            pipe=pipe,
            prompt=prompt,
            seed=seed,
            device=device,
        )

        output_path = (
            OUTPUT_DIR
            / f"batch_{batch_timestamp}_{i:03d}.png"
        )

        save_image(
            image,
            output_path,
        )

        save_image_prompt(
            output_path,
            prompt,
        )

        logs.append(
            {
                "index": i,
                "prompt": prompt,
                "negative_prompt": NEGATIVE_PROMPT,
                "height": HEIGHT,
                "width": WIDTH,
                "num_inference_steps": NUM_INFERENCE_STEPS,
                "guidance_scale": GUIDANCE_SCALE,
                "seed": seed,
                "output": str(output_path),
            }
        )

        print(f"Saved: {output_path}")

    log_path = save_generation_log(
        {
            "script": "batch_generate.py",
            "base_model": BASE_MODEL_NAME,
            "batch_timestamp": batch_timestamp,
            "items": logs,
        }
    )

    print_header("Finished")
    print(f"Log: {log_path}")

    clear_cuda_cache()


if __name__ == "__main__":
    main()
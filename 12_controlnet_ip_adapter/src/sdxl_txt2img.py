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
    PROMPT,
    SEED,
    USE_CUDA_VISIBLE_DEVICES,
    WIDTH,
    create_dirs,
)

if USE_CUDA_VISIBLE_DEVICES:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(
        PHYSICAL_CUDA_ID
    )

# Mac側のOpenMP競合回避用
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


def load_pipeline(
    device: torch.device,
):
    pipe = StableDiffusionXLPipeline.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=(
            torch.float16
            if device.type == "cuda"
            else torch.float32
        ),
        use_safetensors=True,
        variant=(
            "fp16"
            if device.type == "cuda"
            else None
        ),
    )

    pipe.to(device)

    if device.type == "cuda":
        try:
            pipe.enable_xformers_memory_efficient_attention()
            print_header("xFormers enabled.")
        except Exception as e:
            print_header(f"xFormers not enabled: {e}")

    return pipe


def generate_image(
    pipe,
    device: torch.device,
):
    generator = torch.Generator(
        device=device
    ).manual_seed(SEED)

    image = pipe(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        height=HEIGHT,
        width=WIDTH,
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        generator=generator,
    ).images[0]

    return image


def main():

    print_header("SDXL Text-to-Image")

    create_dirs()
    set_seed(SEED)

    device = get_torch_device(
        cuda_id=CUDA_ID
    )

    print_device_info(
        cuda_id=CUDA_ID
    )

    print_header("Prompt")
    print(PROMPT)

    print_header("Negative Prompt")
    print(NEGATIVE_PROMPT)

    pipe = load_pipeline(
        device=device
    )

    image = generate_image(
        pipe=pipe,
        device=device,
    )

    output_path = (
        OUTPUT_DIR
        / f"sdxl_txt2img_{get_timestamp()}.png"
    )

    save_image(
        image,
        output_path,
    )

    save_image_prompt(
        output_path,
        PROMPT,
    )

    log_path = save_generation_log(
        {
            "script": "sdxl_txt2img.py",
            "base_model": BASE_MODEL_NAME,
            "prompt": PROMPT,
            "negative_prompt": NEGATIVE_PROMPT,
            "height": HEIGHT,
            "width": WIDTH,
            "num_inference_steps": NUM_INFERENCE_STEPS,
            "guidance_scale": GUIDANCE_SCALE,
            "seed": SEED,
            "output": str(output_path),
        }
    )

    print_header("Saved")
    print(f"Image: {output_path}")
    print(f"Log  : {log_path}")

    clear_cuda_cache()


if __name__ == "__main__":
    main()
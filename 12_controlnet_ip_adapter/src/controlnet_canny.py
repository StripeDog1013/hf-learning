import os

from config import (
    BASE_MODEL_NAME,
    CONTROLNET_CANNY_MODEL,
    CUDA_ID,
    GUIDANCE_SCALE,
    HEIGHT,
    IMAGE_DIR,
    MAX_IMAGE_SIZE,
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

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from diffusers import (
    ControlNetModel,
    StableDiffusionXLControlNetPipeline,
)

from device import (
    clear_cuda_cache,
    get_torch_device,
    print_device_info,
)
from image_utils import (
    create_canny_image,
    load_image,
    print_image_info,
    resize_keep_aspect,
    save_image,
)
from utils import (
    get_timestamp,
    print_header,
    save_generation_log,
    save_image_prompt,
    set_seed,
)


def get_input_image_path():
    image_path = IMAGE_DIR / "sample.jpg"

    if not image_path.exists():
        raise FileNotFoundError(
            f"Input image not found: {image_path}\n"
            "Please place an image at data/images/sample.jpg"
        )

    return image_path


def load_controlnet(
    device: torch.device,
):
    controlnet = ControlNetModel.from_pretrained(
        CONTROLNET_CANNY_MODEL,
        torch_dtype=(
            torch.float16
            if device.type == "cuda"
            else torch.float32
        ),
        variant=(
            "fp16"
            if device.type == "cuda"
            else None
        ),
        use_safetensors=True,
    )

    controlnet.to(device)

    return controlnet


def load_pipeline(
    device: torch.device,
    controlnet,
):
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        BASE_MODEL_NAME,
        controlnet=controlnet,
        torch_dtype=(
            torch.float16
            if device.type == "cuda"
            else torch.float32
        ),
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

    return pipe


def prepare_control_image():
    image_path = get_input_image_path()

    image = load_image(
        image_path
    )

    print_header("Original Image")
    print_image_info(
        image
    )

    image = resize_keep_aspect(
        image,
        MAX_IMAGE_SIZE,
    )

    # image = image.resize(
    #     (WIDTH, HEIGHT)
    # )

    print_header("Resized Image")
    print_image_info(
        image
    )

    canny_image = create_canny_image(
        image,
        low_threshold=100,
        high_threshold=200,
    )

    print_header("Canny Image")
    print_image_info(
        canny_image
    )

    return image_path, image, canny_image


def generate_image(
    pipe,
    control_image,
    device: torch.device,
):
    generator = torch.Generator(
        device=device
    ).manual_seed(SEED)

    image = pipe(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        image=control_image,
        height=HEIGHT,
        width=WIDTH,
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        controlnet_conditioning_scale=1.0,
        generator=generator,
    ).images[0]

    return image


def main():
    print_header("ControlNet Canny")

    create_dirs()
    set_seed(SEED)

    device = get_torch_device(
        cuda_id=CUDA_ID
    )

    print_device_info(
        cuda_id=CUDA_ID
    )

    input_path, original_image, canny_image = (
        prepare_control_image()
    )

    timestamp = get_timestamp()

    canny_path = (
        OUTPUT_DIR
        / f"controlnet_canny_input_{timestamp}.png"
    )

    save_image(
        canny_image,
        canny_path,
    )

    controlnet = load_controlnet(
        device=device
    )

    pipe = load_pipeline(
        device=device,
        controlnet=controlnet,
    )

    generated_image = generate_image(
        pipe=pipe,
        control_image=canny_image,
        device=device,
    )

    output_path = (
        OUTPUT_DIR
        / f"controlnet_canny_{timestamp}.png"
    )

    save_image(
        generated_image,
        output_path,
    )

    save_image_prompt(
        output_path,
        PROMPT,
    )

    log_path = save_generation_log(
        {
            "script": "controlnet_canny.py",
            "base_model": BASE_MODEL_NAME,
            "controlnet_model": CONTROLNET_CANNY_MODEL,
            "input_image": str(input_path),
            "canny_image": str(canny_path),
            "output_image": str(output_path),
            "prompt": PROMPT,
            "negative_prompt": NEGATIVE_PROMPT,
            "height": HEIGHT,
            "width": WIDTH,
            "num_inference_steps": NUM_INFERENCE_STEPS,
            "guidance_scale": GUIDANCE_SCALE,
            "controlnet_conditioning_scale": 1.0,
            "seed": SEED,
        }
    )

    print_header("Saved")
    print(f"Canny : {canny_path}")
    print(f"Image : {output_path}")
    print(f"Log   : {log_path}")

    clear_cuda_cache()


if __name__ == "__main__":
    main()
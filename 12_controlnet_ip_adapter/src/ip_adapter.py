import os

from config import (
    BASE_MODEL_NAME,
    CUDA_ID,
    GUIDANCE_SCALE,
    HEIGHT,
    IP_ADAPTER_REPO,
    IP_ADAPTER_WEIGHT,
    MAX_IMAGE_SIZE,
    NEGATIVE_PROMPT,
    NUM_INFERENCE_STEPS,
    OUTPUT_DIR,
    PHYSICAL_CUDA_ID,
    PROMPT,
    REFERENCE_DIR,
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
from diffusers import AutoPipelineForText2Image

from device import (
    clear_cuda_cache,
    get_torch_device,
    print_device_info,
)
from image_utils import (
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


IP_ADAPTER_SUBFOLDER = "sdxl_models"
IP_ADAPTER_SCALE = 0.7


def get_reference_image_path():
    image_path = REFERENCE_DIR / "reference.jpg"

    if not image_path.exists():
        raise FileNotFoundError(
            f"Reference image not found: {image_path}\n"
            "Please place an image at "
            "data/reference/reference.jpg"
        )

    return image_path


def load_reference_image():
    image_path = get_reference_image_path()

    image = load_image(
        image_path
    )

    print_header("Reference Image")
    print_image_info(
        image
    )

    image = resize_keep_aspect(
        image,
        MAX_IMAGE_SIZE,
    )

    print_header("Resized Reference Image")
    print_image_info(
        image
    )

    return image_path, image


def load_pipeline(
    device: torch.device,
):
    dtype = (
        torch.float16
        if device.type == "cuda"
        else torch.float32
    )

    pipe = AutoPipelineForText2Image.from_pretrained(
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

    pipe.load_ip_adapter(
        IP_ADAPTER_REPO,
        subfolder=IP_ADAPTER_SUBFOLDER,
        weight_name=IP_ADAPTER_WEIGHT,
    )

    pipe.set_ip_adapter_scale(
        IP_ADAPTER_SCALE
    )

    try:
        pipe.enable_xformers_memory_efficient_attention()
        print_header("xFormers enabled.")
    except Exception as e:
        print_header(f"xFormers not enabled: {e}")

    # 念のため主要コンポーネントを同じdeviceへ揃える
    pipe.text_encoder.to(device)
    pipe.text_encoder_2.to(device)
    pipe.unet.to(device)
    pipe.vae.to(device)

    return pipe


def generate_image(
    pipe,
    reference_image,
    device: torch.device,
):
    generator = torch.Generator(
        device=device
    ).manual_seed(SEED)

    image = pipe(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        ip_adapter_image=reference_image,
        height=HEIGHT,
        width=WIDTH,
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        generator=generator,
    ).images[0]

    return image


def main():

    print_header("IP-Adapter SDXL")

    create_dirs()
    set_seed(SEED)

    device = get_torch_device(
        cuda_id=CUDA_ID
    )

    print_device_info(
        cuda_id=CUDA_ID
    )

    reference_path, reference_image = load_reference_image()

    print_header("Prompt")
    print(PROMPT)

    pipe = load_pipeline(
        device=device
    )

    generated_image = generate_image(
        pipe=pipe,
        reference_image=reference_image,
        device=device,
    )

    timestamp = get_timestamp()

    output_path = (
        OUTPUT_DIR
        / f"ip_adapter_{timestamp}.png"
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
            "script": "ip_adapter.py",
            "base_model": BASE_MODEL_NAME,
            "ip_adapter_repo": IP_ADAPTER_REPO,
            "ip_adapter_subfolder": IP_ADAPTER_SUBFOLDER,
            "ip_adapter_weight": IP_ADAPTER_WEIGHT,
            "ip_adapter_scale": IP_ADAPTER_SCALE,
            "reference_image": str(reference_path),
            "output_image": str(output_path),
            "prompt": PROMPT,
            "negative_prompt": NEGATIVE_PROMPT,
            "height": HEIGHT,
            "width": WIDTH,
            "num_inference_steps": NUM_INFERENCE_STEPS,
            "guidance_scale": GUIDANCE_SCALE,
            "seed": SEED,
        }
    )

    print_header("Saved")
    print(f"Image : {output_path}")
    print(f"Log   : {log_path}")

    clear_cuda_cache()


if __name__ == "__main__":
    main()
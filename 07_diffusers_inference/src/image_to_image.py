import os

from config import (
    USE_CUDA_VISIBLE_DEVICES,
    PHYSICAL_CUDA_ID,
)

if USE_CUDA_VISIBLE_DEVICES:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(PHYSICAL_CUDA_ID)

from pathlib import Path

from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline

from config import (
    MODEL_NAME,
    CUDA_ID,
    PROMPT,
    NEGATIVE_PROMPT,
    NUM_INFERENCE_STEPS,
    GUIDANCE_SCALE,
    WIDTH,
    HEIGHT,
    SEED,
    OUTPUT_DIR,
)

from device import (
    get_device,
    get_torch_dtype,
    print_device_info,
)

from utils import (
    create_generator,
    save_image,
    print_generation_settings,
)


INPUT_IMAGE_PATH = "outputs/generated.png"

OUTPUT_FILE = "image_to_image.png"

STRENGTH = 0.6


def load_image(
    image_path: str,
    width: int,
    height: int,
):
    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Input image not found: {path}"
        )

    image = Image.open(path).convert("RGB")

    image = image.resize(
        (width, height),
        resample=Image.LANCZOS,
    )

    return image


def load_image_to_image_pipeline():
    device = get_device(cuda_id=CUDA_ID)
    torch_dtype = get_torch_dtype(device)

    print(f"Loading img2img pipeline: {MODEL_NAME}")
    print(f"Device: {device}")
    print(f"dtype : {torch_dtype}")

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch_dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )

    pipe = pipe.to(device)

    if device.startswith("cuda"):
        pipe.enable_attention_slicing()

    return pipe


def generate_image_to_image():
    device = get_device(cuda_id=CUDA_ID)

    print_device_info(cuda_id=CUDA_ID)

    print_generation_settings(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        width=WIDTH,
        height=HEIGHT,
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        seed=SEED,
    )

    print(f"Input Image         : {INPUT_IMAGE_PATH}")
    print(f"Strength            : {STRENGTH}")

    init_image = load_image(
        INPUT_IMAGE_PATH,
        WIDTH,
        HEIGHT,
    )

    pipe = load_image_to_image_pipeline()

    generator = create_generator(
        device=device,
        seed=SEED,
    )

    result = pipe(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        image=init_image,
        strength=STRENGTH,
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        generator=generator,
    )

    image = result.images[0]

    output_path = save_image(
        image=image,
        output_dir=OUTPUT_DIR,
        filename=OUTPUT_FILE,
    )

    print("\nImage-to-Image completed.")
    print(f"Output: {output_path}")


def main():
    generate_image_to_image()


if __name__ == "__main__":
    main()
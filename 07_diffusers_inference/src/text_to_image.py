import os

from config import (
    USE_CUDA_VISIBLE_DEVICES,
    PHYSICAL_CUDA_ID,
)

if USE_CUDA_VISIBLE_DEVICES:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(PHYSICAL_CUDA_ID)

from config import (
    CUDA_ID,
    PROMPT,
    NEGATIVE_PROMPT,
    NUM_INFERENCE_STEPS,
    GUIDANCE_SCALE,
    WIDTH,
    HEIGHT,
    SEED,
    OUTPUT_DIR,
    OUTPUT_FILE,
)

from device import (
    get_device,
    print_device_info,
)

from load_pipeline import load_text_to_image_pipeline

from utils import (
    create_generator,
    save_image,
    print_generation_settings,
)


def generate_image():
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

    pipe = load_text_to_image_pipeline()

    generator = create_generator(
        device=device,
        seed=SEED,
    )

    result = pipe(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        width=WIDTH,
        height=HEIGHT,
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

    print("\nImage generation completed.")
    print(f"Output: {output_path}")


def main():
    generate_image()


if __name__ == "__main__":
    main()
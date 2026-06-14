import os

from config import (
    USE_CUDA_VISIBLE_DEVICES,
    PHYSICAL_CUDA_ID,
)

if USE_CUDA_VISIBLE_DEVICES:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(PHYSICAL_CUDA_ID)

from config import (
    CUDA_ID,
    INFERENCE_PROMPT,
    NEGATIVE_PROMPT,
    NUM_INFERENCE_STEPS,
    GUIDANCE_SCALE,
    INFERENCE_SEED,
    SAMPLE_OUTPUT_DIR,
    LORA_OUTPUT_DIR,
)

from device import (
    get_device,
    print_device_info,
)

from load_pipeline import load_lora_pipeline

from utils import (
    create_generator,
    save_image,
    print_inference_settings,
)


OUTPUT_FILE = "lora_inference.png"


def generate_with_lora():
    device = get_device(cuda_id=CUDA_ID)

    print_device_info(cuda_id=CUDA_ID)

    print_inference_settings(
        prompt=INFERENCE_PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        seed=INFERENCE_SEED,
    )

    pipe = load_lora_pipeline(
    lora_path=LORA_OUTPUT_DIR,
    lora_scale=1.5,
    )

    generator = create_generator(
        device=device,
        seed=INFERENCE_SEED,
    )

    result = pipe(
        prompt=INFERENCE_PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        generator=generator,
    )

    image = result.images[0]

    output_path = save_image(
        image=image,
        output_dir=SAMPLE_OUTPUT_DIR,
        filename=OUTPUT_FILE,
    )

    print("\nLoRA inference completed.")
    print(f"Output: {output_path}")


def main():
    generate_with_lora()


if __name__ == "__main__":
    main()
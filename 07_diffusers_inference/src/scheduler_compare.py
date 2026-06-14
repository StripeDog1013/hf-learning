import os

from config import (
    USE_CUDA_VISIBLE_DEVICES,
    PHYSICAL_CUDA_ID,
)

if USE_CUDA_VISIBLE_DEVICES:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(PHYSICAL_CUDA_ID)

from diffusers import (
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    DPMSolverMultistepScheduler,
)

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


SCHEDULERS = {
    "euler": EulerDiscreteScheduler,
    "euler_a": EulerAncestralDiscreteScheduler,
    "dpm_solver": DPMSolverMultistepScheduler,
}


def set_scheduler(pipe, scheduler_class):
    pipe.scheduler = scheduler_class.from_config(
        pipe.scheduler.config,
    )

    return pipe


def generate_with_scheduler(
    pipe,
    scheduler_name: str,
    scheduler_class,
):
    device = get_device(cuda_id=CUDA_ID)

    pipe = set_scheduler(
        pipe,
        scheduler_class,
    )

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
        filename=f"scheduler_{scheduler_name}.png",
    )

    print(
        f"Saved {scheduler_name}: "
        f"{output_path}"
    )


def main():
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

    for scheduler_name, scheduler_class in SCHEDULERS.items():
        print(f"\n=== Scheduler: {scheduler_name} ===")

        generate_with_scheduler(
            pipe=pipe,
            scheduler_name=scheduler_name,
            scheduler_class=scheduler_class,
        )

    print("\nScheduler comparison completed.")


if __name__ == "__main__":
    main()
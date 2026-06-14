import os
import time

from config import (
    USE_CUDA_VISIBLE_DEVICES,
    PHYSICAL_CUDA_ID,
)

if USE_CUDA_VISIBLE_DEVICES:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(PHYSICAL_CUDA_ID)

import torch

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


NUM_RUNS = 5
WARMUP_RUNS = 1


def sync_device(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def generate_once(pipe, device: str, seed: int):
    generator = create_generator(
        device=device,
        seed=seed,
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

    return result.images[0]


def main():
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

    print("\n=== Warmup ===")
    for i in range(WARMUP_RUNS):
        _ = generate_once(
            pipe=pipe,
            device=device,
            seed=SEED + i,
        )
        sync_device(device)
        print(f"Warmup {i + 1}/{WARMUP_RUNS} completed")

    times = []
    last_image = None

    print("\n=== Benchmark ===")

    for i in range(NUM_RUNS):
        sync_device(device)

        start_time = time.perf_counter()

        last_image = generate_once(
            pipe=pipe,
            device=device,
            seed=SEED + i,
        )

        sync_device(device)

        elapsed = time.perf_counter() - start_time

        times.append(elapsed)

        print(
            f"Run {i + 1}/{NUM_RUNS}: "
            f"{elapsed:.3f} sec"
        )

    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)

    output_path = save_image(
        image=last_image,
        output_dir=OUTPUT_DIR,
        filename="benchmark_last.png",
    )

    print("\n=== Benchmark Result ===")
    print(f"Runs       : {NUM_RUNS}")
    print(f"Average    : {avg_time:.3f} sec/image")
    print(f"Min        : {min_time:.3f} sec/image")
    print(f"Max        : {max_time:.3f} sec/image")
    print(f"Last Image : {output_path}")


if __name__ == "__main__":
    main()
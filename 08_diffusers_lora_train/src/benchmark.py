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

from load_pipeline import (
    load_base_pipeline,
    load_lora_pipeline,
)

from utils import (
    create_generator,
    save_image,
    print_inference_settings,
)


NUM_RUNS = 5
WARMUP_RUNS = 1


def sync_device(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def generate_once(
    pipe,
    device: str,
    seed: int,
):
    generator = create_generator(
        device=device,
        seed=seed,
    )

    result = pipe(
        prompt=INFERENCE_PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        generator=generator,
    )

    return result.images[0]


def benchmark_pipeline(
    name: str,
    pipe,
    device: str,
):
    print(f"\n=== Benchmark: {name} ===")

    for i in range(WARMUP_RUNS):
        _ = generate_once(
            pipe=pipe,
            device=device,
            seed=INFERENCE_SEED + i,
        )

        sync_device(device)

        print(
            f"Warmup {i + 1}/{WARMUP_RUNS} completed"
        )

    times = []
    last_image = None

    for i in range(NUM_RUNS):
        sync_device(device)

        start_time = time.perf_counter()

        last_image = generate_once(
            pipe=pipe,
            device=device,
            seed=INFERENCE_SEED + i,
        )

        sync_device(device)

        elapsed = (
            time.perf_counter()
            - start_time
        )

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
        output_dir=SAMPLE_OUTPUT_DIR,
        filename=f"benchmark_{name}.png",
    )

    print(f"\n{name} Result")
    print(f"Average    : {avg_time:.3f} sec/image")
    print(f"Min        : {min_time:.3f} sec/image")
    print(f"Max        : {max_time:.3f} sec/image")
    print(f"Last Image : {output_path}")

    return {
        "name": name,
        "average": avg_time,
        "min": min_time,
        "max": max_time,
    }


def main():
    device = get_device(cuda_id=CUDA_ID)

    print_device_info(cuda_id=CUDA_ID)

    print_inference_settings(
        prompt=INFERENCE_PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        seed=INFERENCE_SEED,
    )

    results = []

    base_pipe = load_base_pipeline()

    results.append(
        benchmark_pipeline(
            name="base",
            pipe=base_pipe,
            device=device,
        )
    )

    del base_pipe

    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    lora_pipe = load_lora_pipeline(
        lora_path=LORA_OUTPUT_DIR,
        lora_scale=1.0,
    )

    results.append(
        benchmark_pipeline(
            name="lora",
            pipe=lora_pipe,
            device=device,
        )
    )

    print("\n=== Benchmark Summary ===")

    for result in results:
        print(
            f"{result['name']:>6}: "
            f"{result['average']:.3f} sec/image "
            f"(min={result['min']:.3f}, "
            f"max={result['max']:.3f})"
        )


if __name__ == "__main__":
    main()
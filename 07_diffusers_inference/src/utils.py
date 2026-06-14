from pathlib import Path

import torch


def create_directory(path: str | Path) -> None:
    Path(path).mkdir(
        parents=True,
        exist_ok=True,
    )


def create_generator(
    device: str,
    seed: int,
):
    if device.startswith("cuda"):
        return torch.Generator(
            device=device,
        ).manual_seed(seed)

    # MPS / CPU は CPU Generator の方が安定
    return torch.Generator(
        device="cpu",
    ).manual_seed(seed)


def save_image(
    image,
    output_dir: str | Path,
    filename: str,
) -> Path:
    output_dir = Path(output_dir)

    create_directory(output_dir)

    output_path = output_dir / filename

    image.save(output_path)

    return output_path


def print_generation_settings(
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    num_inference_steps: int,
    guidance_scale: float,
    seed: int,
) -> None:
    print("\n=== Generation Settings ===")
    print(f"Prompt              : {prompt}")
    print(f"Negative Prompt     : {negative_prompt}")
    print(f"Size                : {width} x {height}")
    print(f"Steps               : {num_inference_steps}")
    print(f"Guidance Scale      : {guidance_scale}")
    print(f"Seed                : {seed}")


def main():
    print("=== utils.py Test ===")

    create_directory("outputs/utils_test")
    print("Directory Test: OK")

    generator = create_generator(
        device="cpu",
        seed=42,
    )

    value = torch.rand(
        1,
        generator=generator,
    )

    print(f"Seed Test Value: {value.item():.6f}")

    print_generation_settings(
        prompt="a cute shiba inu",
        negative_prompt="low quality",
        width=512,
        height=512,
        num_inference_steps=4,
        guidance_scale=0.0,
        seed=42,
    )

    print("\n=== Test Complete ===")


if __name__ == "__main__":
    main()
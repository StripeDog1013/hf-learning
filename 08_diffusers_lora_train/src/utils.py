from pathlib import Path

import torch


def create_directory(path: str | Path) -> None:
    Path(path).mkdir(
        parents=True,
        exist_ok=True,
    )


def initialize_seed(seed: int) -> None:
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def create_generator(
    device: str,
    seed: int,
):
    if device.startswith("cuda"):
        return torch.Generator(
            device=device,
        ).manual_seed(seed)

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


def print_train_settings(
    model_name: str,
    data_dir,
    image_size: int,
    batch_size: int,
    learning_rate: float,
    num_epochs: int,
    lora_rank: int,
    lora_alpha: int,
) -> None:
    print("\n=== Training Settings ===")
    print(f"Model Name     : {model_name}")
    print(f"Data Dir       : {data_dir}")
    print(f"Image Size     : {image_size}")
    print(f"Batch Size     : {batch_size}")
    print(f"Learning Rate  : {learning_rate}")
    print(f"Epochs         : {num_epochs}")
    print(f"LoRA Rank      : {lora_rank}")
    print(f"LoRA Alpha     : {lora_alpha}")


def print_inference_settings(
    prompt: str,
    negative_prompt: str,
    num_inference_steps: int,
    guidance_scale: float,
    seed: int,
) -> None:
    print("\n=== Inference Settings ===")
    print(f"Prompt          : {prompt}")
    print(f"Negative Prompt : {negative_prompt}")
    print(f"Steps           : {num_inference_steps}")
    print(f"Guidance Scale  : {guidance_scale}")
    print(f"Seed            : {seed}")


def main():
    print("=== utils.py Test ===")

    create_directory("outputs/utils_test")
    print("Directory Test: OK")

    initialize_seed(42)

    generator = create_generator(
        device="cpu",
        seed=42,
    )

    value = torch.rand(
        1,
        generator=generator,
    )

    print(f"Seed Test Value: {value.item():.6f}")

    print("\n=== Test Complete ===")


if __name__ == "__main__":
    main()
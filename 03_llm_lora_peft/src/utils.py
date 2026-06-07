from pathlib import Path

import torch

from transformers import set_seed


def create_directory(path: str) -> None:
    Path(path).mkdir(
        parents=True,
        exist_ok=True,
    )


def initialize_seed(seed: int) -> None:
    set_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def count_trainable_parameters(model) -> tuple[int, int]:
    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    return trainable_params, total_params


def print_model_info(model) -> None:
    trainable_params, total_params = (
        count_trainable_parameters(model)
    )

    trainable_ratio = (
        100 * trainable_params / total_params
        if total_params > 0
        else 0
    )

    print("\n=== Model Information ===")
    print(f"Trainable Parameters : {trainable_params:,}")
    print(f"Total Parameters     : {total_params:,}")
    print(f"Trainable Ratio      : {trainable_ratio:.4f}%")


def print_lora_status(model) -> None:
    print("\n=== LoRA Status ===")

    if hasattr(model, "peft_config"):
        print("PEFT Model : True")
        print(f"Adapters   : {list(model.peft_config.keys())}")
    else:
        print("PEFT Model : False")


def main() -> None:
    print("=== utils.py Test ===")

    print("\n[1] create_directory()")

    test_dir = "logs/utils_test"
    create_directory(test_dir)

    print(f"Created: {test_dir}")
    print(f"Exists : {Path(test_dir).exists()}")

    print("\n[2] initialize_seed()")

    initialize_seed(42)
    x = torch.rand(3)

    initialize_seed(42)
    y = torch.rand(3)

    print("Tensor X:", x)
    print("Tensor Y:", y)
    print("Seed Test:", torch.equal(x, y))

    print("\n[3] print_model_info()")

    model = torch.nn.Sequential(
        torch.nn.Linear(10, 20),
        torch.nn.ReLU(),
        torch.nn.Linear(20, 1),
    )

    print_model_info(model)

    print("\n[4] print_lora_status()")
    print_lora_status(model)

    print("\n=== Test Complete ===")


if __name__ == "__main__":
    main()
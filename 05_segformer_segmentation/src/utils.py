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
    trainable_params, total_params = count_trainable_parameters(model)

    ratio = (
        100 * trainable_params / total_params
        if total_params > 0
        else 0
    )

    print("\n=== Model Information ===")
    print(f"Trainable Parameters : {trainable_params:,}")
    print(f"Total Parameters     : {total_params:,}")
    print(f"Trainable Ratio      : {ratio:.4f}%")


def main() -> None:
    print("=== utils.py Test ===")

    create_directory("logs/utils_test")
    print("Directory Test: OK")

    initialize_seed(42)
    x = torch.rand(3)

    initialize_seed(42)
    y = torch.rand(3)

    print("Seed Test:", torch.equal(x, y))

    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 16, kernel_size=3),
        torch.nn.ReLU(),
        torch.nn.Conv2d(16, 150, kernel_size=1),
    )

    print_model_info(model)

    print("\n=== Test Complete ===")


if __name__ == "__main__":
    main()
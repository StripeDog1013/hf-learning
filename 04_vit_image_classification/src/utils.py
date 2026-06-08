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


def count_trainable_parameters(model):
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

    print(
        f"Trainable Parameters : "
        f"{trainable_params:,}"
    )

    print(
        f"Total Parameters     : "
        f"{total_params:,}"
    )

    print(
        f"Trainable Ratio      : "
        f"{trainable_ratio:.4f}%"
    )


def main():

    print("=== utils.py Test ===")

    print("\n[1] create_directory()")

    test_dir = "./logs/utils_test"

    create_directory(test_dir)

    print(f"Created: {test_dir}")

    print(
        f"Exists : "
        f"{Path(test_dir).exists()}"
    )

    print("\n[2] initialize_seed()")

    initialize_seed(42)

    x = torch.rand(3)

    initialize_seed(42)

    y = torch.rand(3)

    print("Tensor X:", x)

    print("Tensor Y:", y)

    print(
        "Seed Test:",
        torch.equal(x, y)
    )

    print(
        "\n[3] count_trainable_parameters()"
    )

    model = torch.nn.Sequential(
        torch.nn.Linear(10, 20),
        torch.nn.ReLU(),
        torch.nn.Linear(20, 3),
    )

    trainable, total = (
        count_trainable_parameters(model)
    )

    print(
        f"Trainable: "
        f"{trainable:,}"
    )

    print(
        f"Total    : "
        f"{total:,}"
    )

    print("\n[4] print_model_info()")

    print_model_info(model)

    print("\n=== Test Complete ===")


if __name__ == "__main__":
    main()
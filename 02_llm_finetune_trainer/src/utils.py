from pathlib import Path

import torch

from transformers import set_seed


def create_directory(path: str) -> None:
    """
    ディレクトリ作成
    """

    Path(path).mkdir(
        parents=True,
        exist_ok=True,
    )


def initialize_seed(seed: int) -> None:
    """
    乱数シード固定
    """

    set_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def count_trainable_parameters(model) -> tuple[int, int]:
    """
    学習対象パラメータ数取得

    Returns
    -------
    trainable_params
    total_params
    """

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
    """
    モデル情報表示
    """

    trainable_params, total_params = (
        count_trainable_parameters(model)
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
        f"{100 * trainable_params / total_params:.2f}%"
    )
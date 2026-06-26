import json

import random

from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from config import OUTPUT_DIR


def print_header(
    title: str,
):

    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def get_timestamp():

    return datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )


def set_seed(
    seed: int,
):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)


def save_image_prompt(
    image_path: Path,
    prompt: str,
):

    txt_path = image_path.with_suffix(
        ".txt"
    )

    with open(
        txt_path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(prompt)


def save_json(
    data,
    file_path: Path,
):

    file_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        file_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False,
        )


def save_generation_log(
    log_data,
):

    filename = (
        OUTPUT_DIR
        / (
            "generation_log_"
            + get_timestamp()
            + ".json"
        )
    )

    save_json(
        log_data,
        filename,
    )

    return filename
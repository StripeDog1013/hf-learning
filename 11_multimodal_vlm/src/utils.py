import json

from datetime import datetime
from pathlib import Path

from config import (
    OUTPUT_DIR,
)


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


def load_json(
    file_path: Path,
):

    with open(
        file_path,
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


def save_text(
    text: str,
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

        f.write(text)


def save_chat_log(
    messages,
    prefix="chat",
):

    filename = (
        OUTPUT_DIR
        / f"{prefix}_{get_timestamp()}.json"
    )

    save_json(
        messages,
        filename,
    )

    return filename


if __name__ == "__main__":

    print_header(
        "utils.py Test"
    )

    print(
        get_timestamp()
    )
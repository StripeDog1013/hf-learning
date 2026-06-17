import json
from datetime import datetime
from pathlib import Path

from config import (
    CHECKPOINT_DIR,
    DATA_DIR,
    MODEL_DIR,
    OUTPUT_DIR,
)


def create_dirs():

    dirs = [
        DATA_DIR,
        MODEL_DIR,
        OUTPUT_DIR,
        CHECKPOINT_DIR,
    ]

    for dir_path in dirs:

        dir_path.mkdir(
            parents=True,
            exist_ok=True,
        )


def print_header(
    title: str,
):

    line = "=" * 60

    print(f"\n{line}")
    print(title)
    print(line)


def get_timestamp() -> str:

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

    with file_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
        )


def load_json(
    file_path: Path,
):

    with file_path.open(
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

    with file_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        f.write(text)


def load_text(
    file_path: Path,
) -> str:

    with file_path.open(
        "r",
        encoding="utf-8",
    ) as f:

        return f.read()


def save_chat_log(
    messages: list[dict],
    prefix: str = "chat",
) -> Path:

    timestamp = get_timestamp()

    file_path = (
        OUTPUT_DIR
        / f"{prefix}_{timestamp}.json"
    )

    save_json(
        messages,
        file_path,
    )

    return file_path


def format_messages(
    messages: list[dict],
) -> str:

    lines = []

    for message in messages:

        role = message.get(
            "role",
            "unknown",
        )

        content = message.get(
            "content",
            "",
        )

        lines.append(
            f"[{role}]\n{content}"
        )

    return "\n\n".join(lines)


if __name__ == "__main__":

    create_dirs()

    print_header("utils.py Test")

    sample_messages = [
        {
            "role": "user",
            "content": "現在時刻を教えて",
        },
        {
            "role": "assistant",
            "content": "ツールを使って確認します。",
        },
    ]

    print(
        format_messages(sample_messages)
    )

    log_path = save_chat_log(
        sample_messages,
        prefix="utils_test",
    )

    print(f"\nSaved log: {log_path}")

    print("\nutils.py test finished.")
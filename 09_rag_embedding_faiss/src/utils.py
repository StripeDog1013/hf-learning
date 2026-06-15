import json
from pathlib import Path

from config import (
    CHECKPOINT_DIR,
    DATA_DIR,
    DOCUMENT_DIR,
    INDEX_DIR,
    MODEL_DIR,
    OUTPUT_DIR,
    SUPPORTED_EXTENSIONS,
)


def create_dirs():
    dirs = [
        DATA_DIR,
        DOCUMENT_DIR,
        MODEL_DIR,
        OUTPUT_DIR,
        CHECKPOINT_DIR,
        INDEX_DIR,
    ]

    for dir_path in dirs:
        dir_path.mkdir(
            parents=True,
            exist_ok=True,
        )


def load_text_file(
    file_path: Path,
) -> str:
    with file_path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return f.read()


def load_documents(
    document_dir: Path = DOCUMENT_DIR,
) -> list[dict]:

    documents = []

    for file_path in sorted(
        document_dir.rglob("*")
    ):
        if not file_path.is_file():
            continue

        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        text = load_text_file(file_path)

        documents.append(
            {
                "source": str(file_path),
                "text": text,
            }
        )

    return documents


def split_text(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:

    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be greater than 0"
        )

    if chunk_overlap < 0:
        raise ValueError(
            "chunk_overlap must be 0 or greater"
        )

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be smaller than chunk_size"
        )

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - chunk_overlap

    return chunks


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


def print_header(
    title: str,
):

    line = "=" * 60

    print(f"\n{line}")
    print(title)
    print(line)


if __name__ == "__main__":

    create_dirs()

    print_header("utils.py Test")

    documents = load_documents()

    print(
        f"Loaded documents: "
        f"{len(documents)}"
    )

    sample_text = (
        "RAGは検索拡張生成です。"
        "Embeddingで文章をベクトル化し、"
        "FAISSで類似検索します。"
    )

    chunks = split_text(
        text=sample_text,
        chunk_size=20,
        chunk_overlap=5,
    )

    print(
        f"Sample chunks: "
        f"{len(chunks)}"
    )

    for i, chunk in enumerate(chunks):
        print(f"[{i}] {chunk}")

    sample_json_path = (
        OUTPUT_DIR / "utils_test.json"
    )

    save_json(
        {
            "message": "utils.py test",
            "chunks": chunks,
        },
        sample_json_path,
    )

    loaded = load_json(sample_json_path)

    print(
        f"Loaded JSON message: "
        f"{loaded['message']}"
    )

    print("\nutils.py test finished.")
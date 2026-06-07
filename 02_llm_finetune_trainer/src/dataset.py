from datasets import load_dataset

from config import (
    TRAIN_FILE,
    VALID_FILE,
)


def load_train_dataset():
    """
    学習データセット読込
    """

    dataset = load_dataset(
        "json",
        data_files=TRAIN_FILE,
        split="train",
    )

    return dataset


def load_valid_dataset():
    """
    検証データセット読込
    """

    dataset = load_dataset(
        "json",
        data_files=VALID_FILE,
        split="train",
    )

    return dataset


def load_datasets():
    """
    学習・検証データセット読込
    """

    train_dataset = load_train_dataset()
    valid_dataset = load_valid_dataset()

    return train_dataset, valid_dataset


def print_dataset_info(dataset, title="Dataset"):
    """
    データセット情報表示
    """

    print(f"\n=== {title} ===")

    print(f"Rows    : {len(dataset)}")
    print(f"Columns : {dataset.column_names}")

    if len(dataset) > 0:
        print("\nFirst Sample")
        print(dataset[0])

if __name__ == "__main__":

    train_dataset, valid_dataset = load_datasets()

    print_dataset_info(
        train_dataset,
        "Train Dataset",
    )

    print_dataset_info(
        valid_dataset,
        "Validation Dataset",
    )
from datasets import load_dataset

from config import (
    TRAIN_FILE,
    VALID_FILE,
)


def load_train_dataset():

    return load_dataset(
        "json",
        data_files=TRAIN_FILE,
        split="train",
    )


def load_valid_dataset():

    return load_dataset(
        "json",
        data_files=VALID_FILE,
        split="train",
    )


def load_datasets():

    train_dataset = load_train_dataset()

    valid_dataset = load_valid_dataset()

    return train_dataset, valid_dataset


def print_dataset_info(
    dataset,
    title="Dataset",
):

    print(f"\n=== {title} ===")

    print(f"Rows    : {len(dataset)}")

    print(
        f"Columns : "
        f"{dataset.column_names}"
    )

    if len(dataset) > 0:

        print("\nFirst Sample")

        print(dataset[0])
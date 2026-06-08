from datasets import load_dataset

from config import (
    DATASET_NAME,
    LOCAL_IMAGE_DATA_DIR,
    USE_LOCAL_IMAGE_FOLDER,
)


def load_datasets():
    if USE_LOCAL_IMAGE_FOLDER:
        dataset = load_dataset(
            "imagefolder",
            data_dir=LOCAL_IMAGE_DATA_DIR,
        )
    else:
        dataset = load_dataset(
            DATASET_NAME,
        )

    train_dataset = dataset["train"]
    valid_dataset = dataset["validation"]

    return train_dataset, valid_dataset


def print_dataset_info(
    dataset,
    title="Dataset",
):
    print(f"\n=== {title} ===")
    print(f"Rows    : {len(dataset)}")
    print(f"Columns : {dataset.column_names}")

    if len(dataset) > 0:
        print("\nFirst Sample")
        print(dataset[0])


def get_label_names(dataset):
    label_names = get_label_column_name(dataset)
    label_feature = dataset.features[label_names]

    return label_feature.names

def get_label_column_name(dataset) -> str:
    if "labels" in dataset.column_names:
        return "labels"

    if "label" in dataset.column_names:
        return "label"

    raise ValueError(
        f"Label column not found. columns={dataset.column_names}"
    )

def print_label_info(train_dataset):
    label_names = get_label_names(train_dataset)

    print("\n=== Labels ===")

    for idx, label in enumerate(label_names):
        print(f"{idx}: {label}")


def main():
    train_dataset, valid_dataset = load_datasets()

    print_dataset_info(
        train_dataset,
        "Train Dataset",
    )

    print_dataset_info(
        valid_dataset,
        "Validation Dataset",
    )

    print_label_info(train_dataset)


if __name__ == "__main__":
    main()
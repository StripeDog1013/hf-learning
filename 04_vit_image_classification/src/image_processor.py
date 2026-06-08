from transformers import AutoImageProcessor

from config import MODEL_NAME
from dataset import get_label_column_name


def load_image_processor():
    print(f"Loading image processor: {MODEL_NAME}")

    image_processor = AutoImageProcessor.from_pretrained(
        MODEL_NAME,
    )

    return image_processor


def transform_example(example, image_processor, label_column: str):
    image = example["image"]

    inputs = image_processor(
        image,
        return_tensors="pt",
    )

    example["pixel_values"] = inputs["pixel_values"][0]
    example["labels"] = example[label_column]

    return example


def transform_dataset(dataset, image_processor):
    label_column = get_label_column_name(dataset)

    transformed_dataset = dataset.map(
        lambda example: transform_example(
            example,
            image_processor,
            label_column,
        ),
        remove_columns=[
            col
            for col in dataset.column_names
            if col != label_column
        ],
    )

    if label_column != "labels":
        transformed_dataset = transformed_dataset.remove_columns(
            [label_column]
        )

    return transformed_dataset


def print_processed_sample(dataset, index: int = 0):
    sample = dataset[index]

    print("\n=== Processed Sample ===")
    print(f"Keys               : {sample.keys()}")
    print(f"pixel_values shape : {sample['pixel_values'].shape}")
    print(f"labels             : {sample['labels']}")


def main():
    from dataset import load_datasets

    image_processor = load_image_processor()

    train_dataset, _ = load_datasets()

    processed_dataset = transform_dataset(
        train_dataset,
        image_processor,
    )

    processed_dataset.set_format(type="torch")

    print_processed_sample(
        processed_dataset,
    )


if __name__ == "__main__":
    main()
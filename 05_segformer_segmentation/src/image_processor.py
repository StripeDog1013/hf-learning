import torch

from torch.utils.data import Dataset
from transformers import SegformerImageProcessor

from config import (
    MODEL_NAME,
    IMAGE_SIZE,
)


class SegmentationProcessedDataset(Dataset):
    def __init__(self, dataset, image_processor):
        self.dataset = dataset
        self.image_processor = image_processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int):
        example = self.dataset[index]

        image = example["image"]
        annotation = example["annotation"]

        inputs = self.image_processor(
            images=image,
            segmentation_maps=annotation,
            return_tensors="pt",
        )

        labels = inputs["labels"][0]

        labels = torch.where(
        labels == 150,
        torch.tensor(
            255,
            dtype=labels.dtype,
            device=labels.device,
            ),
            labels,
        )

        return {
            "pixel_values": inputs["pixel_values"][0],
            "labels": labels,
        }


def load_image_processor():
    print(f"Loading image processor: {MODEL_NAME}")

    image_processor = SegformerImageProcessor.from_pretrained(
        MODEL_NAME,
        size={
            "height": IMAGE_SIZE,
            "width": IMAGE_SIZE,
        },
    )

    return image_processor


def transform_dataset(dataset, image_processor):
    return SegmentationProcessedDataset(
        dataset,
        image_processor,
    )


def print_processed_sample(dataset, index: int = 0):
    sample = dataset[index]

    pixel_values = sample["pixel_values"]
    labels = sample["labels"]

    print("\n=== Processed Sample ===")
    print(f"Keys               : {sample.keys()}")
    print(f"pixel_values type  : {type(pixel_values)}")
    print(f"labels type        : {type(labels)}")
    print(f"pixel_values shape : {pixel_values.shape}")
    print(f"labels shape       : {labels.shape}")
    print(f"labels min/max     : {labels.min()} / {labels.max()}")


def main():
    from dataset import load_datasets

    image_processor = load_image_processor()

    train_dataset, _ = load_datasets()

    processed_dataset = transform_dataset(
        train_dataset,
        image_processor,
    )

    print_processed_sample(processed_dataset, 10)


if __name__ == "__main__":
    main()
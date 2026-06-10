import json
from pathlib import Path

from PIL import Image
from datasets import load_dataset

from config import (
    DATASET_NAME,
    USE_LOCAL_IMAGE_FOLDER,
    LOCAL_DATA_DIR,
)


def load_hub_datasets():
    dataset = load_dataset(DATASET_NAME)

    train_dataset = dataset["train"]
    valid_dataset = dataset["test"]

    return train_dataset, valid_dataset


class LocalObjectDetectionDataset:
    def __init__(self, split: str):
        self.image_dir = Path(LOCAL_DATA_DIR) / split / "images"
        self.annotation_dir = Path(LOCAL_DATA_DIR) / split / "annotations"

        self.annotation_paths = sorted(
            self.annotation_dir.glob("*.json")
        )

        if len(self.annotation_paths) == 0:
            raise FileNotFoundError(
                f"No annotation files found in: {self.annotation_dir}"
            )

    def __len__(self):
        return len(self.annotation_paths)

    def __getitem__(self, index: int):
        annotation_path = self.annotation_paths[index]

        with open(annotation_path, "r", encoding="utf-8") as f:
            annotation = json.load(f)

        image_id = annotation["image_id"]
        image_path = self.image_dir / f"{image_id}.jpg"

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image file not found: {image_path}"
            )

        image = Image.open(image_path).convert("RGB")

        return {
            "image_id": image_id,
            "image": image,
            "width": annotation["width"],
            "height": annotation["height"],
            "objects": annotation["objects"],
        }


def load_local_datasets():
    train_dataset = LocalObjectDetectionDataset("train")
    valid_dataset = LocalObjectDetectionDataset("validation")

    return train_dataset, valid_dataset


def load_datasets():
    if USE_LOCAL_IMAGE_FOLDER:
        return load_local_datasets()

    return load_hub_datasets()


def print_dataset_info(dataset, title="Dataset"):
    print(f"\n=== {title} ===")
    print(f"Rows : {len(dataset)}")

    sample = dataset[0]

    print(f"Keys : {sample.keys()}")

    image = sample["image"]
    objects = sample["objects"]

    print("\n=== Sample ===")
    print(f"Image Type   : {type(image)}")
    print(f"Image Size   : {image.size}")
    print(f"Image ID     : {sample['image_id']}")
    print(f"Width/Height : {sample['width']} / {sample['height']}")
    print(f"Object Count : {len(objects['bbox'])}")
    print(f"Categories   : {objects['category']}")
    print(f"BBoxes       : {objects['bbox']}")


def main():
    train_dataset, valid_dataset = load_datasets()

    print_dataset_info(train_dataset, "Train Dataset")
    print_dataset_info(valid_dataset, "Validation Dataset")


if __name__ == "__main__":
    main()
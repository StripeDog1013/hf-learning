from pathlib import Path

from PIL import Image
from datasets import load_dataset
from torch.utils.data import Dataset

from config import (
    DATASET_NAME,
    USE_LOCAL_IMAGE_FOLDER,
    LOCAL_DATA_DIR,
)


class LocalSegmentationDataset(Dataset):
    def __init__(self, split: str):
        self.image_dir = Path(LOCAL_DATA_DIR) / split / "images"
        self.mask_dir = Path(LOCAL_DATA_DIR) / split / "masks"

        self.image_paths = sorted(self.image_dir.glob("*.jpg"))

        if len(self.image_paths) == 0:
            raise FileNotFoundError(
                f"No images found in: {self.image_dir}"
            )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index: int):
        image_path = self.image_paths[index]
        mask_path = self.mask_dir / f"{image_path.stem}.png"
        if not mask_path.exists():
            raise FileNotFoundError(
                f"Mask not found: {mask_path}"
            )

        image = Image.open(image_path).convert("RGB")
        annotation = Image.open(mask_path)

        return {
            "image": image,
            "annotation": annotation,
        }


def load_datasets():
    if USE_LOCAL_IMAGE_FOLDER:
        train_dataset = LocalSegmentationDataset("train")
        valid_dataset = LocalSegmentationDataset("validation")
    else:
        dataset = load_dataset(DATASET_NAME)

        train_dataset = dataset["train"]
        valid_dataset = dataset["validation"]

    return train_dataset, valid_dataset


def print_dataset_info(dataset, title="Dataset"):
    print(f"\n=== {title} ===")
    print(f"Rows : {len(dataset)}")

    sample = dataset[0]

    print("\nFirst Sample Keys")
    print(sample.keys())

    print("\n=== Sample ===")
    print(f"Image Type : {type(sample['image'])}")
    print(f"Mask Type  : {type(sample['annotation'])}")
    print(f"Image Size : {sample['image'].size}")
    print(f"Mask Size  : {sample['annotation'].size}")


def main():
    train_dataset, valid_dataset = load_datasets()

    print_dataset_info(train_dataset, "Train Dataset")
    print_dataset_info(valid_dataset, "Validation Dataset")


if __name__ == "__main__":
    main()
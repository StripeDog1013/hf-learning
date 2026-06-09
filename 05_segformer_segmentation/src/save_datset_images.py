from pathlib import Path

from datasets import load_dataset

from config import DATASET_NAME


OUTPUT_ROOT = Path("data/scene_parse_150")


def save_split(dataset, split_name: str) -> None:
    output_image_dir = OUTPUT_ROOT / split_name / "images"
    output_mask_dir = OUTPUT_ROOT / split_name / "masks"

    output_image_dir.mkdir(parents=True, exist_ok=True)
    output_mask_dir.mkdir(parents=True, exist_ok=True)

    for i, sample in enumerate(dataset):
        image = sample["image"]
        mask = sample["annotation"]

        image_path = output_image_dir / f"{i:06d}.jpg"
        mask_path = output_mask_dir / f"{i:06d}.png"

        image.convert("RGB").save(image_path)
        mask.save(mask_path)

        if (i + 1) % 1000 == 0:
            print(f"{split_name}: saved {i + 1} samples")


def main() -> None:
    dataset = load_dataset(DATASET_NAME)

    save_split(dataset["train"], "train")
    save_split(dataset["validation"], "validation")

    print("\nDataset images and masks saved.")
    print(f"Output: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
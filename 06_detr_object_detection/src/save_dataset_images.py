import json
from pathlib import Path

from dataset import load_datasets


OUTPUT_DIR = Path("data/cppe-5")


def save_split(
    dataset,
    split_name: str,
):
    image_dir = (
        OUTPUT_DIR
        / split_name
        / "images"
    )

    annotation_dir = (
        OUTPUT_DIR
        / split_name
        / "annotations"
    )

    image_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    annotation_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    total = len(dataset)

    print(
        f"\nSaving {split_name}"
    )

    for index in range(total):
        sample = dataset[index]

        image = sample["image"].convert("RGB")

        image_id = sample["image_id"]

        image_path = (
            image_dir
            / f"{image_id}.jpg"
        )

        image.save(image_path)

        annotation = {
            "image_id": sample["image_id"],
            "width": sample["width"],
            "height": sample["height"],
            "objects": sample["objects"],
        }

        annotation_path = (
            annotation_dir
            / f"{image_id}.json"
        )

        with open(
            annotation_path,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                annotation,
                f,
                ensure_ascii=False,
                indent=2,
            )

        if (index + 1) % 100 == 0:
            print(
                f"{index + 1}/{total}"
            )

    print(
        f"Completed {split_name}"
    )


def main():
    train_dataset, valid_dataset = (
        load_datasets()
    )

    save_split(
        train_dataset,
        "train",
    )

    save_split(
        valid_dataset,
        "validation",
    )


if __name__ == "__main__":
    main()
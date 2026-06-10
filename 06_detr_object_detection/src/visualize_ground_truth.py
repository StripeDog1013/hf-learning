from pathlib import Path

from PIL import ImageDraw, ImageFont

from dataset import load_datasets
from load_model import ID2LABEL


# "train" or "validation"
TARGET_SPLIT = "validation"

# Noneなら全件、整数ならその件数だけ保存
MAX_SAMPLES = None

OUTPUT_DIR = Path("logs/ground_truth")


def get_dataset():
    train_dataset, valid_dataset = load_datasets()

    if TARGET_SPLIT == "train":
        return train_dataset

    if TARGET_SPLIT == "validation":
        return valid_dataset

    raise ValueError(
        "TARGET_SPLIT must be 'train' or 'validation'"
    )


def draw_box(
    draw,
    bbox,
    label_name: str,
):
    x, y, w, h = bbox

    x1 = x
    y1 = y
    x2 = x + w
    y2 = y + h

    draw.rectangle(
        [x1, y1, x2, y2],
        outline="red",
        width=3,
    )

    text = label_name

    draw.rectangle(
        [x1, y1, x1 + 140, y1 + 20],
        fill="red",
    )

    draw.text(
        (x1 + 4, y1 + 2),
        text,
        fill="white",
    )


def visualize_sample(
    sample,
    output_path: Path,
):
    image = sample["image"].convert("RGB")

    objects = sample["objects"]

    draw = ImageDraw.Draw(image)

    for bbox, category in zip(
        objects["bbox"],
        objects["category"],
    ):
        label_name = ID2LABEL.get(
            int(category),
            str(category),
        )

        draw_box(
            draw,
            bbox,
            label_name,
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    image.save(output_path)


def main():
    dataset = get_dataset()

    total = len(dataset)

    if MAX_SAMPLES is not None:
        total = min(
            total,
            MAX_SAMPLES,
        )

    output_dir = (
        OUTPUT_DIR
        / TARGET_SPLIT
    )

    print("\n=== Visualize Ground Truth ===")
    print(f"Split      : {TARGET_SPLIT}")
    print(f"Total      : {total}")
    print(f"Output Dir : {output_dir}")

    for index in range(total):
        sample = dataset[index]

        output_path = (
            output_dir
            / f"{index:06d}.jpg"
        )

        visualize_sample(
            sample,
            output_path,
        )

        if (index + 1) % 10 == 0:
            print(
                f"Saved {index + 1}/{total}"
            )

    print("\nCompleted.")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
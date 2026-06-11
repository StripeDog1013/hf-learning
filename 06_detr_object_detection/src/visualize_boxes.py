import os

from config import USE_CUDA_VISIBLE_DEVICES, PHYSICAL_CUDA_ID

if USE_CUDA_VISIBLE_DEVICES:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(PHYSICAL_CUDA_ID)

from pathlib import Path

import torch
from PIL import ImageDraw

from config import (
    CUDA_ID,
    OUTPUT_DIR,
)

from device import get_device, print_device_info
from dataset import load_datasets
from image_processor import load_image_processor
from load_model import (
    load_model_for_inference,
    ID2LABEL,
)


MODEL_PATH = f"{OUTPUT_DIR}/final_model"

# "train" or "validation"
# TARGET_SPLIT = "train"
TARGET_SPLIT = "validation"

SCORE_THRESHOLD = 0.50

# Noneなら全件、整数なら件数制限
MAX_SAMPLES = None # 20

OUTPUT_ROOT = Path("logs/predicted_boxes")


def get_dataset():
    train_dataset, valid_dataset = load_datasets()

    if TARGET_SPLIT == "train":
        return train_dataset

    if TARGET_SPLIT == "validation":
        return valid_dataset

    raise ValueError(
        "TARGET_SPLIT must be 'train' or 'validation'"
    )


def predict(
    image,
    image_processor,
    model,
    device,
):
    inputs = image_processor(
        images=image,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor(
        [image.size[::-1]],
        device=device,
    )

    results = (
        image_processor
        .post_process_object_detection(
            outputs,
            threshold=SCORE_THRESHOLD,
            target_sizes=target_sizes,
        )[0]
    )

    return results


def draw_predictions(
    image,
    results,
):
    draw = ImageDraw.Draw(image)

    boxes = results["boxes"]
    scores = results["scores"]
    labels = results["labels"]

    for box, score, label in zip(
        boxes,
        scores,
        labels,
    ):
        x1, y1, x2, y2 = box.tolist()

        label_id = int(label.item())

        label_name = ID2LABEL.get(
            label_id,
            str(label_id),
        )

        text = (
            f"{label_name}: "
            f"{score.item():.2f}"
        )

        draw.rectangle(
            [x1, y1, x2, y2],
            outline="red",
            width=3,
        )

        text_width = max(
            90,
            len(text) * 8,
        )

        draw.rectangle(
            [
                x1,
                y1,
                x1 + text_width,
                y1 + 20,
            ],
            fill="red",
        )

        draw.text(
            (x1 + 4, y1 + 3),
            text,
            fill="white",
        )

    return image


def visualize_sample(
    sample,
    image_processor,
    model,
    device,
    output_path: Path,
):
    image = sample["image"].convert("RGB")

    results = predict(
        image=image,
        image_processor=image_processor,
        model=model,
        device=device,
    )

    output_image = draw_predictions(
        image.copy(),
        results,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_image.save(output_path)

    return len(results["scores"])


def main():
    print_device_info(cuda_id=CUDA_ID)

    device = get_device(cuda_id=CUDA_ID)

    dataset = get_dataset()

    image_processor = load_image_processor()

    model = load_model_for_inference(
        MODEL_PATH,
        cuda_id=CUDA_ID,
    )

    total = len(dataset)

    if MAX_SAMPLES is not None:
        total = min(total, MAX_SAMPLES)

    output_dir = OUTPUT_ROOT / TARGET_SPLIT

    print("\n=== Visualize Predicted Boxes ===")
    print(f"Split      : {TARGET_SPLIT}")
    print(f"Total      : {total}")
    print(f"Threshold  : {SCORE_THRESHOLD}")
    print(f"Output Dir : {output_dir}")

    for index in range(total):
        sample = dataset[index]

        output_path = (
            output_dir
            / f"{index:06d}.jpg"
        )

        detected_count = visualize_sample(
            sample=sample,
            image_processor=image_processor,
            model=model,
            device=device,
            output_path=output_path,
        )

        print(
            f"[{index + 1}/{total}] "
            f"detected={detected_count} "
            f"saved={output_path}"
        )

    print("\nCompleted.")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
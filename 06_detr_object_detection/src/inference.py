import os

from config import USE_CUDA_VISIBLE_DEVICES, PHYSICAL_CUDA_ID

if USE_CUDA_VISIBLE_DEVICES:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(PHYSICAL_CUDA_ID)

from pathlib import Path

import torch
from PIL import Image, ImageDraw

from config import (
    CUDA_ID,
    OUTPUT_DIR,
)

from device import get_device, print_device_info
from dataset import load_datasets
from image_processor import load_image_processor
from load_model import load_model_for_inference


MODEL_PATH = f"{OUTPUT_DIR}/final_model"

SAMPLE_INDEX = 10

SCORE_THRESHOLD = 0.5

OUTPUT_PATH = "logs/inference_result.jpg"


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

        label_name = str(label.item())

        draw.rectangle(
            [x1, y1, x2, y2],
            outline="red",
            width=3,
        )

        text = f"{label_name}: {score.item():.2f}"

        draw.text(
            (x1, y1),
            text,
            fill="red",
        )

    return image


def predict(image):
    device = get_device(cuda_id=CUDA_ID)

    image_processor = load_image_processor()

    model = load_model_for_inference(
        MODEL_PATH,
        cuda_id=CUDA_ID,
    )

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

    results = image_processor.post_process_object_detection(
        outputs,
        threshold=SCORE_THRESHOLD,
        target_sizes=target_sizes,
    )[0]

    return results


def main():
    print_device_info(cuda_id=CUDA_ID)

    _, valid_dataset = load_datasets()

    sample = valid_dataset[SAMPLE_INDEX]

    image = sample["image"].convert("RGB")

    results = predict(image)

    output_image = draw_predictions(
        image.copy(),
        results,
    )

    output_path = Path(OUTPUT_PATH)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_image.save(output_path)

    print("\n=== Inference Result ===")
    print(f"Detected objects: {len(results['scores'])}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
import os

from config import USE_CUDA_VISIBLE_DEVICES, PHYSICAL_CUDA_ID

if USE_CUDA_VISIBLE_DEVICES:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(PHYSICAL_CUDA_ID)

import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from config import CUDA_ID, OUTPUT_DIR
from device import get_device, print_device_info
from dataset import load_datasets
from image_processor import load_image_processor
from load_model import load_model_for_inference


MODEL_PATH = f"{OUTPUT_DIR}/final_model"

SCORE_THRESHOLD = 0.0


def xywh_to_xyxy(boxes):
    converted = []

    for x, y, w, h in boxes:
        converted.append(
            [
                x,
                y,
                x + w,
                y + h,
            ]
        )

    return torch.tensor(
        converted,
        dtype=torch.float32,
    )


def build_target(sample):
    objects = sample["objects"]

    boxes = xywh_to_xyxy(
        objects["bbox"]
    )

    labels = torch.tensor(
        objects["category"],
        dtype=torch.long,
    )

    return {
        "boxes": boxes,
        "labels": labels,
    }


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

    result = image_processor.post_process_object_detection(
        outputs,
        threshold=SCORE_THRESHOLD,
        target_sizes=target_sizes,
    )[0]

    return {
        "boxes": result["boxes"].cpu(),
        "scores": result["scores"].cpu(),
        "labels": result["labels"].cpu(),
    }


def main():
    print_device_info(cuda_id=CUDA_ID)

    device = get_device(cuda_id=CUDA_ID)

    _, valid_dataset = load_datasets()

    image_processor = load_image_processor()

    model = load_model_for_inference(
        MODEL_PATH,
        cuda_id=CUDA_ID,
    )

    metric = MeanAveragePrecision(
        box_format="xyxy",
        iou_type="bbox",
    )

    total = len(valid_dataset)

    print("\n=== mAP Evaluation ===")
    print(f"Samples: {total}")

    for index in range(total):
        sample = valid_dataset[index]

        image = sample["image"].convert("RGB")

        prediction = predict(
            image=image,
            image_processor=image_processor,
            model=model,
            device=device,
        )

        target = build_target(sample)

        metric.update(
            [prediction],
            [target],
        )

        if (index + 1) % 10 == 0:
            print(f"Processed {index + 1}/{total}")

    results = metric.compute()

    print("\n=== mAP Result ===")

    for key, value in results.items():
        if torch.is_tensor(value):
            print(f"{key}: {value.item() if value.numel() == 1 else value}")


if __name__ == "__main__":
    main()
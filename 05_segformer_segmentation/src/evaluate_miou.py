import os

from config import USE_CUDA_VISIBLE_DEVICES, PHYSICAL_CUDA_ID

if USE_CUDA_VISIBLE_DEVICES:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(PHYSICAL_CUDA_ID)

import torch

from config import CUDA_ID, OUTPUT_DIR
from device import get_device, print_device_info
from dataset import load_datasets
from image_processor import load_image_processor, transform_dataset
from load_model import load_model_for_inference


MODEL_PATH = f"{OUTPUT_DIR}/final_model"

NUM_CLASSES = 150
IGNORE_INDEX = 255


def predict_mask(model, pixel_values, target_size):
    with torch.no_grad():
        outputs = model(
            pixel_values=pixel_values,
        )

    logits = outputs.logits

    upsampled_logits = torch.nn.functional.interpolate(
        logits,
        size=target_size,
        mode="bilinear",
        align_corners=False,
    )

    predictions = upsampled_logits.argmax(dim=1)

    return predictions


def update_intersection_union(
    pred,
    label,
    intersections,
    unions,
):
    valid_mask = label != IGNORE_INDEX

    pred = pred[valid_mask]
    label = label[valid_mask]

    for class_id in range(NUM_CLASSES):
        pred_mask = pred == class_id
        label_mask = label == class_id

        intersection = (pred_mask & label_mask).sum().item()
        union = (pred_mask | label_mask).sum().item()

        intersections[class_id] += intersection
        unions[class_id] += union


def compute_miou(intersections, unions):
    ious = []

    for class_id in range(NUM_CLASSES):
        if unions[class_id] == 0:
            continue

        iou = intersections[class_id] / unions[class_id]
        ious.append(iou)

    miou = sum(ious) / len(ious) if len(ious) > 0 else 0.0

    return miou, ious


def main():
    print_device_info(cuda_id=CUDA_ID)

    device = get_device(cuda_id=CUDA_ID)

    _, valid_dataset = load_datasets()

    image_processor = load_image_processor()

    valid_dataset = transform_dataset(
        valid_dataset,
        image_processor,
    )

    model = load_model_for_inference(
        MODEL_PATH,
        cuda_id=CUDA_ID,
    )

    intersections = [0 for _ in range(NUM_CLASSES)]
    unions = [0 for _ in range(NUM_CLASSES)]

    for index in range(len(valid_dataset)):
        sample = valid_dataset[index]

        pixel_values = sample["pixel_values"].unsqueeze(0).to(device)
        labels = sample["labels"].to(device)

        pred = predict_mask(
            model=model,
            pixel_values=pixel_values,
            target_size=labels.shape[-2:],
        )[0]

        update_intersection_union(
            pred=pred,
            label=labels,
            intersections=intersections,
            unions=unions,
        )

        if (index + 1) % 100 == 0:
            print(f"Processed {index + 1}/{len(valid_dataset)}")

    miou, ious = compute_miou(
        intersections,
        unions,
    )

    print("\n=== mIoU Result ===")
    print(f"Evaluated Classes : {len(ious)}")
    print(f"mIoU              : {miou:.4f}")


if __name__ == "__main__":
    main()
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

    correct = 0
    total = 0

    for index in range(len(valid_dataset)):
        sample = valid_dataset[index]

        pixel_values = sample["pixel_values"].unsqueeze(0).to(device)
        labels = sample["labels"].to(device)

        pred = predict_mask(
            model=model,
            pixel_values=pixel_values,
            target_size=labels.shape[-2:],
        )[0]

        valid_mask = labels != 255

        correct += (pred[valid_mask] == labels[valid_mask]).sum().item()
        total += valid_mask.sum().item()

        if (index + 1) % 100 == 0:
            print(f"Processed {index + 1}/{len(valid_dataset)}")

    pixel_accuracy = correct / total if total > 0 else 0.0

    print("\n=== Pixel Accuracy Result ===")
    print(f"Correct        : {correct}")
    print(f"Total Pixels   : {total}")
    print(f"Pixel Accuracy : {pixel_accuracy:.4f}")


if __name__ == "__main__":
    main()
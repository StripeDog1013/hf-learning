import os

from config import (
    USE_CUDA_VISIBLE_DEVICES,
    PHYSICAL_CUDA_ID,
)

if USE_CUDA_VISIBLE_DEVICES:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(PHYSICAL_CUDA_ID)

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from config import (
    CUDA_ID,
    OUTPUT_DIR,
)

from device import get_device, print_device_info
from dataset import load_datasets
from image_processor import load_image_processor
from load_model import load_model_for_inference


MODEL_PATH = f"{OUTPUT_DIR}/final_model"

SAMPLE_INDEX = 0

OUTPUT_MASK_PATH = "logs/pred_mask.png"


def create_color_palette(num_classes: int = 150) -> np.ndarray:
    rng = np.random.default_rng(seed=42)

    return rng.integers(
        low=0,
        high=255,
        size=(num_classes, 3),
        dtype=np.uint8,
    )


def mask_to_color_image(mask: np.ndarray) -> Image.Image:
    palette = create_color_palette()

    color_mask = np.zeros(
        (mask.shape[0], mask.shape[1], 3),
        dtype=np.uint8,
    )

    valid_area = mask != 255

    color_mask[valid_area] = palette[
        mask[valid_area]
    ]

    return Image.fromarray(color_mask)


def predict(image: Image.Image):
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

    logits = outputs.logits

    upsampled_logits = torch.nn.functional.interpolate(
        logits,
        size=image.size[::-1],
        mode="bilinear",
        align_corners=False,
    )

    pred_mask = upsampled_logits.argmax(dim=1)[0]

    return pred_mask.cpu().numpy()


def main():
    print_device_info(cuda_id=CUDA_ID)

    _, valid_dataset = load_datasets()

    sample = valid_dataset[SAMPLE_INDEX]

    image = sample["image"].convert("RGB")

    pred_mask = predict(image)

    pred_mask_img = mask_to_color_image(pred_mask)

    output_path = Path(OUTPUT_MASK_PATH)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pred_mask_img.save(output_path)

    print("\nInference completed.")
    print(f"Output mask: {output_path}")


if __name__ == "__main__":
    main()
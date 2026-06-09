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

# "train" or "validation"
TARGET_SPLIT = "validation"

# Noneなら全件、整数ならその件数だけ保存
MAX_SAMPLES = None

OUTPUT_ROOT = "logs/visualizations"


def create_color_palette(
    num_classes: int = 150,
) -> np.ndarray:
    rng = np.random.default_rng(seed=42)

    return rng.integers(
        low=0,
        high=255,
        size=(num_classes, 3),
        dtype=np.uint8,
    )


def mask_to_color_image(
    mask: np.ndarray,
    palette: np.ndarray,
) -> Image.Image:
    h, w = mask.shape

    color_mask = np.zeros(
        (h, w, 3),
        dtype=np.uint8,
    )

    valid_area = (
        (mask != 255)
        & (mask >= 0)
        & (mask < len(palette))
    )

    color_mask[valid_area] = palette[
        mask[valid_area]
    ]

    return Image.fromarray(color_mask)


def normalize_gt_mask(
    mask: Image.Image,
) -> np.ndarray:
    mask_array = np.array(mask)

    # 今回のADE20Kでは150が混じることがあるためignoreへ
    mask_array = np.where(
        mask_array == 150,
        255,
        mask_array,
    )

    return mask_array


def prepare_inputs(
    image: Image.Image,
    image_processor,
    device: str,
):
    inputs = image_processor(
        images=image,
        return_tensors="pt",
    )

    return {
        key: value.to(device)
        for key, value in inputs.items()
    }


def predict_mask(
    image: Image.Image,
    image_processor,
    model,
    device: str,
) -> np.ndarray:
    inputs = prepare_inputs(
        image,
        image_processor,
        device,
    )

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits

    upsampled_logits = torch.nn.functional.interpolate(
        logits,
        size=image.size[::-1],
        mode="bilinear",
        align_corners=False,
    )

    pred_mask = upsampled_logits.argmax(
        dim=1,
    )[0]

    return pred_mask.cpu().numpy()


def make_canvas(
    image: Image.Image,
    gt_mask_img: Image.Image,
    pred_mask_img: Image.Image,
) -> Image.Image:
    width, height = image.size

    canvas = Image.new(
        "RGB",
        (width * 3, height),
        color=(255, 255, 255),
    )

    canvas.paste(
        image.convert("RGB"),
        (0, 0),
    )

    canvas.paste(
        gt_mask_img.convert("RGB"),
        (width, 0),
    )

    canvas.paste(
        pred_mask_img.convert("RGB"),
        (width * 2, 0),
    )

    return canvas


def get_target_dataset():
    train_dataset, valid_dataset = load_datasets()

    if TARGET_SPLIT == "train":
        return train_dataset

    if TARGET_SPLIT == "validation":
        return valid_dataset

    raise ValueError(
        "TARGET_SPLIT must be 'train' or 'validation'"
    )


def visualize_sample(
    sample,
    image_processor,
    model,
    device: str,
    palette: np.ndarray,
    output_path: Path,
) -> None:
    image = sample["image"].convert("RGB")
    gt_mask = sample["annotation"]

    pred_mask = predict_mask(
        image=image,
        image_processor=image_processor,
        model=model,
        device=device,
    )

    gt_mask_array = normalize_gt_mask(
        gt_mask,
    )

    gt_mask_img = mask_to_color_image(
        gt_mask_array,
        palette,
    )

    pred_mask_img = mask_to_color_image(
        pred_mask,
        palette,
    )

    canvas = make_canvas(
        image=image,
        gt_mask_img=gt_mask_img,
        pred_mask_img=pred_mask_img,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    canvas.save(output_path)


def main():
    print_device_info(cuda_id=CUDA_ID)

    device = get_device(cuda_id=CUDA_ID)

    dataset = get_target_dataset()

    image_processor = load_image_processor()

    model = load_model_for_inference(
        MODEL_PATH,
        cuda_id=CUDA_ID,
    )

    palette = create_color_palette()

    output_dir = (
        Path(OUTPUT_ROOT)
        / TARGET_SPLIT
    )

    total = len(dataset)

    if MAX_SAMPLES is not None:
        total = min(
            total,
            MAX_SAMPLES,
        )

    print("\n=== Visualization Settings ===")
    print(f"Split       : {TARGET_SPLIT}")
    print(f"Total       : {total}")
    print(f"Output Dir  : {output_dir}")

    for index in range(total):
        sample = dataset[index]

        output_path = (
            output_dir
            / f"{index:06d}.png"
        )

        visualize_sample(
            sample=sample,
            image_processor=image_processor,
            model=model,
            device=device,
            palette=palette,
            output_path=output_path,
        )

        if (index + 1) % 10 == 0:
            print(
                f"Saved {index + 1}/{total}"
            )

    print("\nVisualization completed.")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
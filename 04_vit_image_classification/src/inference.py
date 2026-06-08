from pathlib import Path

import torch
from PIL import Image
from transformers import (
    AutoImageProcessor,
    ViTForImageClassification,
)

from config import (
    CUDA_ID,
    OUTPUT_DIR,
)

from device import get_device, print_device_info


MODEL_PATH = f"{OUTPUT_DIR}/final_model"

# save_dataset_images.py で保存した画像から1枚指定
IMAGE_PATH = "data/images/validation/healthy/00089.jpg"


def load_model_and_processor():
    device = get_device(cuda_id=CUDA_ID)

    print(f"Loading model: {MODEL_PATH}")

    image_processor = AutoImageProcessor.from_pretrained(
        MODEL_PATH,
    )

    model = ViTForImageClassification.from_pretrained(
        MODEL_PATH,
    )

    model.to(device)
    model.eval()

    return image_processor, model


def load_image(image_path: str):
    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Image file not found: {image_path}"
        )

    image = Image.open(path).convert("RGB")

    return image


def predict(image_path: str):
    device = get_device(cuda_id=CUDA_ID)

    image_processor, model = load_model_and_processor()

    image = load_image(image_path)

    inputs = image_processor(
        image,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits

    predicted_id = logits.argmax(dim=-1).item()

    predicted_label = model.config.id2label[predicted_id]

    probabilities = torch.softmax(
        logits,
        dim=-1,
    )[0]

    confidence = probabilities[predicted_id].item()

    return predicted_label, confidence


def main():
    print_device_info(cuda_id=CUDA_ID)

    print("\n=== Image ===")
    print(IMAGE_PATH)

    predicted_label, confidence = predict(
        IMAGE_PATH,
    )

    print("\n=== Prediction ===")
    print(f"Label      : {predicted_label}")
    print(f"Confidence : {confidence:.4f}")


if __name__ == "__main__":
    main()
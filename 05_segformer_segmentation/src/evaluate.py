import os

from config import (
    USE_CUDA_VISIBLE_DEVICES,
    PHYSICAL_CUDA_ID,
)

if USE_CUDA_VISIBLE_DEVICES:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(PHYSICAL_CUDA_ID)

import numpy as np
import torch
from transformers import Trainer, TrainingArguments

from config import (
    CUDA_ID,
    EVAL_BATCH_SIZE,
    OUTPUT_DIR,
    LOG_DIR,
    SEED,
)

from device import print_device_info
from dataset import load_datasets
from image_processor import load_image_processor, transform_dataset
from load_model import load_model_for_inference


MODEL_PATH = f"{OUTPUT_DIR}/final_model"


def compute_metrics(eval_pred):
    logits, labels = eval_pred

    predictions = np.argmax(logits, axis=1)

    if predictions.shape[-2:] != labels.shape[-2:]:
        # Trainer側で既に揃っていることが多いが、念のため
        return {}

    valid_mask = labels != 255

    correct = (
        predictions[valid_mask]
        == labels[valid_mask]
    ).sum()

    total = valid_mask.sum()

    pixel_accuracy = correct / total if total > 0 else 0.0

    return {
        "pixel_accuracy": float(pixel_accuracy),
    }


def build_eval_args():
    return TrainingArguments(
        output_dir=OUTPUT_DIR,
        logging_dir=LOG_DIR,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,#BATCH_SIZE,
        fp16=torch.cuda.is_available(),
        report_to="none",
        seed=SEED,
        remove_unused_columns=False,
        eval_accumulation_steps=1,
    )


def main():
    print_device_info(cuda_id=CUDA_ID)

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

    trainer = Trainer(
        model=model,
        args=build_eval_args(),
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
    )

    metrics = trainer.evaluate()

    print("\n=== Evaluation Result ===")
    print(metrics)

    if "eval_pixel_accuracy" in metrics:
        print(
            f"\nPixel Accuracy: "
            f"{metrics['eval_pixel_accuracy']:.4f}"
        )


if __name__ == "__main__":
    main()
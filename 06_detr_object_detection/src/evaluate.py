import os

from config import USE_CUDA_VISIBLE_DEVICES, PHYSICAL_CUDA_ID

if USE_CUDA_VISIBLE_DEVICES:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(PHYSICAL_CUDA_ID)

import torch
from transformers import Trainer, TrainingArguments

from config import (
    CUDA_ID,
    BATCH_SIZE,
    OUTPUT_DIR,
    LOG_DIR,
    SEED,
)

from device import print_device_info
from dataset import load_datasets, print_dataset_info
from image_processor import (
    load_image_processor,
    transform_dataset,
    collate_fn,
)
from load_model import load_model_for_inference


MODEL_PATH = f"{OUTPUT_DIR}/final_model"


def build_eval_args():
    return TrainingArguments(
        output_dir=OUTPUT_DIR,
        logging_dir=LOG_DIR,
        per_device_eval_batch_size=BATCH_SIZE,
        fp16=torch.cuda.is_available(),
        report_to="none",
        seed=SEED,
        remove_unused_columns=False,
        eval_accumulation_steps=1,
    )


def main():
    print_device_info(cuda_id=CUDA_ID)

    _, valid_dataset = load_datasets()

    print_dataset_info(
        valid_dataset,
        "Validation Dataset",
    )

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
        data_collator=collate_fn,
    )

    metrics = trainer.evaluate()

    print("\n=== Evaluation Result ===")
    print(metrics)

    if "eval_loss" in metrics:
        print(f"\nEval Loss: {metrics['eval_loss']:.4f}")


if __name__ == "__main__":
    main()
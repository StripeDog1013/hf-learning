import os

from config import (
    USE_CUDA_VISIBLE_DEVICES,
    PHYSICAL_CUDA_ID,
)

if USE_CUDA_VISIBLE_DEVICES:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(PHYSICAL_CUDA_ID)

import torch

from transformers import (
    Trainer,
    TrainingArguments,
)

from config import (
    CUDA_ID,
    NUM_EPOCHS,
    BATCH_SIZE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    WARMUP_RATIO,
    LOGGING_STEPS,
    SAVE_STEPS,
    EVAL_STEPS,
    OUTPUT_DIR,
    LOG_DIR,
    SAVE_TOTAL_LIMIT,
    LOAD_BEST_MODEL_AT_END,
    EVALUATION_STRATEGY,
    SAVE_STRATEGY,
    SEED,
)

from device import print_device_info

from dataset import (
    load_datasets,
    print_dataset_info,
)

from image_processor import (
    load_image_processor,
    transform_dataset,
)

from load_model import load_model_for_inference

from utils import initialize_seed


BASE_MODEL_PATH = f"{OUTPUT_DIR}/final_model"
CONTINUED_MODEL_PATH = f"{OUTPUT_DIR}/continued_model"


def build_training_args():
    return TrainingArguments(
        output_dir=OUTPUT_DIR,
        logging_dir=LOG_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        eval_steps=EVAL_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        eval_strategy=EVALUATION_STRATEGY,
        save_strategy=SAVE_STRATEGY,
        load_best_model_at_end=LOAD_BEST_MODEL_AT_END,
        fp16=torch.cuda.is_available(),
        report_to="none",
        seed=SEED,
        remove_unused_columns=False,
    )


def main():
    initialize_seed(SEED)

    print_device_info(cuda_id=CUDA_ID)

    train_dataset, valid_dataset = load_datasets()

    print_dataset_info(train_dataset, "Train Dataset")

    image_processor = load_image_processor()

    model = load_model_for_inference(
        BASE_MODEL_PATH,
        cuda_id=CUDA_ID,
    )

    train_dataset = transform_dataset(
        train_dataset,
        image_processor,
    )

    valid_dataset = transform_dataset(
        valid_dataset,
        image_processor,
    )

    train_dataset.set_format(type="torch")
    valid_dataset.set_format(type="torch")

    training_args = build_training_args()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
    )

    sample = train_dataset[0]
    print("========================================")
    print(sample.keys())
    print(sample["pixel_values"].shape)
    print(sample["labels"])
    print(next(model.parameters()).device)
    print("========================================")
    trainer.train()

    trainer.save_model(CONTINUED_MODEL_PATH)
    image_processor.save_pretrained(CONTINUED_MODEL_PATH)

    print("\nContinue training completed.")
    print(f"Model saved to: {CONTINUED_MODEL_PATH}")


if __name__ == "__main__":
    main()
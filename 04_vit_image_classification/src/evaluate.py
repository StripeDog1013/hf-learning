import numpy as np
import torch

from transformers import (
    Trainer,
    TrainingArguments,
)

from config import (
    CUDA_ID,
    BATCH_SIZE,
    OUTPUT_DIR,
    LOG_DIR,
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


MODEL_PATH = f"{OUTPUT_DIR}/final_model"
# MODEL_PATH = "checkpoints/continued_model"


def compute_metrics(eval_pred):
    logits, labels = eval_pred

    predictions = np.argmax(
        logits,
        axis=-1,
    )

    accuracy = (
        predictions == labels
    ).mean()

    return {
        "accuracy": accuracy,
    }


def build_eval_args():
    return TrainingArguments(
        output_dir=OUTPUT_DIR,
        logging_dir=LOG_DIR,
        per_device_eval_batch_size=BATCH_SIZE,
        fp16=torch.cuda.is_available(),
        report_to="none",
        seed=SEED,
    )


def main():
    print_device_info(
        cuda_id=CUDA_ID,
    )

    _, valid_dataset = load_datasets()

    print_dataset_info(
        valid_dataset,
        "Validation Dataset",
    )

    # image_processor = load_image_processor()
    from transformers import AutoImageProcessor

    image_processor = AutoImageProcessor.from_pretrained(MODEL_PATH)

    valid_dataset = transform_dataset(
        valid_dataset,
        image_processor,
    )

    valid_dataset.set_format(
        type="torch",
    )

    model = load_model_for_inference(
        MODEL_PATH,
        cuda_id=CUDA_ID,
    )
    
    eval_args = build_eval_args()

    trainer = Trainer(
        model=model,
        args=eval_args,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
    )

    metrics = trainer.evaluate()

    print("\n=== Evaluation Result ===")
    print(metrics)

    if "eval_accuracy" in metrics:
        print(
            f"\nAccuracy: "
            f"{metrics['eval_accuracy']:.4f}"
        )


if __name__ == "__main__":
    main()
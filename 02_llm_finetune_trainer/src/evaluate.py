import math
import torch

from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

from config import (
    CUDA_ID,
    BATCH_SIZE,
    OUTPUT_DIR,
    LOG_DIR,
    SEED,
)

from device import print_device_info
from load_model import load_tokenizer, load_model
from dataset import load_valid_dataset, print_dataset_info
from tokenizer_utils import (
    tokenize_dataset,
    print_tokenized_sample,
)


def build_eval_args() -> TrainingArguments:
    return TrainingArguments(
        output_dir=OUTPUT_DIR,
        logging_dir=LOG_DIR,
        per_device_eval_batch_size=BATCH_SIZE,
        fp16=torch.cuda.is_available(),
        report_to="none",
        seed=SEED,
    )


def main() -> None:
    print_device_info(cuda_id=CUDA_ID)

    tokenizer = load_tokenizer()
    model = load_model()

    valid_dataset = load_valid_dataset()

    print_dataset_info(
        valid_dataset,
        "Validation Dataset",
    )

    tokenized_valid_dataset = tokenize_dataset(
        valid_dataset,
        tokenizer,
    )

    print_tokenized_sample(
        tokenized_valid_dataset,
        tokenizer,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    eval_args = build_eval_args()

    trainer = Trainer(
        model=model,
        args=eval_args,
        eval_dataset=tokenized_valid_dataset,
        data_collator=data_collator,
    )

    metrics = trainer.evaluate()

    eval_loss = metrics.get("eval_loss")

    print("\n=== Evaluation Result ===")
    print(metrics)

    if eval_loss is not None:
        perplexity = math.exp(eval_loss)

        print(f"\nEval loss  : {eval_loss:.4f}")
        print(f"Perplexity : {perplexity:.4f}")


if __name__ == "__main__":
    main()
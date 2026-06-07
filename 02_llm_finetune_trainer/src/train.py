import torch

from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    set_seed,
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
from load_model import load_llm
from dataset import load_datasets, print_dataset_info
from tokenizer_utils import (
    tokenize_dataset,
    print_tokenized_sample,
)


def build_training_args() -> TrainingArguments:
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
    )


def main() -> None:
    set_seed(SEED)

    print_device_info(cuda_id=CUDA_ID)

    tokenizer, model = load_llm()

    train_dataset, valid_dataset = load_datasets()

    print_dataset_info(train_dataset, "Train Dataset")
    print_dataset_info(valid_dataset, "Validation Dataset")

    tokenized_train_dataset = tokenize_dataset(
        train_dataset,
        tokenizer,
    )

    tokenized_valid_dataset = tokenize_dataset(
        valid_dataset,
        tokenizer,
    )

    print_tokenized_sample(
        tokenized_train_dataset,
        tokenizer,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    training_args = build_training_args()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_valid_dataset,
        data_collator=data_collator,
    )

    trainer.train()

    trainer.save_model(f"{OUTPUT_DIR}/final_model")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/final_model")

    print("\nTraining completed.")
    print(f"Final model saved to: {OUTPUT_DIR}/final_model")


if __name__ == "__main__":
    main()
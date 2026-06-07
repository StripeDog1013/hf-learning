import torch

from peft import get_peft_model

from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
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
from lora_config import (
    build_lora_config,
    print_lora_config,
)
from utils import (
    initialize_seed,
    print_model_info,
    print_lora_status,
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
    initialize_seed(SEED)

    print_device_info(cuda_id=CUDA_ID)

    tokenizer, base_model = load_llm()

    lora_config = build_lora_config()

    print_lora_config(lora_config)

    model = get_peft_model(
        base_model,
        lora_config,
    )

    print("\n=== LoRA Model ===")
    print_model_info(model)
    print_lora_status(model)
    model.print_trainable_parameters()

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

    print("\nResume LoRA training from latest checkpoint...")

    trainer.train(
        resume_from_checkpoint=True,
    )

    final_lora_dir = f"{OUTPUT_DIR}/final_lora"

    model.save_pretrained(final_lora_dir)
    tokenizer.save_pretrained(final_lora_dir)

    print("\nResume LoRA training completed.")
    print(f"Final LoRA adapter saved to: {final_lora_dir}")


if __name__ == "__main__":
    main()
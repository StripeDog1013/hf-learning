import math
import torch

from peft import PeftModel

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

from config import (
    CUDA_ID,
    MODEL_NAME,
    OUTPUT_DIR,
    LOG_DIR,
    BATCH_SIZE,
    TRUST_REMOTE_CODE,
    SEED,
)

from device import get_device, print_device_info
from dataset import load_valid_dataset, print_dataset_info
from tokenizer_utils import (
    tokenize_dataset,
    print_tokenized_sample,
)
from utils import print_model_info, print_lora_status


LORA_PATH = f"{OUTPUT_DIR}/final_lora"


def load_lora_model():
    device = get_device(cuda_id=CUDA_ID)

    print(f"Loading base model: {MODEL_NAME}")
    print(f"Loading LoRA adapter: {LORA_PATH}")

    tokenizer = AutoTokenizer.from_pretrained(
        LORA_PATH,
        trust_remote_code=TRUST_REMOTE_CODE,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=TRUST_REMOTE_CODE,
        dtype=torch.float32,
    )

    model = PeftModel.from_pretrained(
        base_model,
        LORA_PATH,
    )

    model.to(device)
    model.eval()

    return tokenizer, model


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

    tokenizer, model = load_lora_model()

    print_model_info(model)
    print_lora_status(model)

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

    print("\n=== LoRA Evaluation Result ===")
    print(metrics)

    if eval_loss is not None:
        perplexity = math.exp(eval_loss)

        print(f"\nEval loss  : {eval_loss:.4f}")
        print(f"Perplexity : {perplexity:.4f}")


if __name__ == "__main__":
    main()
import torch

from peft import PeftModel

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from config import (
    CUDA_ID,
    MODEL_NAME,
    OUTPUT_DIR,
    TRUST_REMOTE_CODE,
)

from device import get_device, print_device_info
from utils import print_model_info, print_lora_status


LORA_PATH = f"{OUTPUT_DIR}/final_lora"

MERGED_MODEL_PATH = f"{OUTPUT_DIR}/merged_model"


def load_base_model():
    print(f"Loading base model: {MODEL_NAME}")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=TRUST_REMOTE_CODE,
        dtype=torch.float32,
    )

    return model


def load_tokenizer():
    print(f"Loading tokenizer: {LORA_PATH}")

    tokenizer = AutoTokenizer.from_pretrained(
        LORA_PATH,
        trust_remote_code=TRUST_REMOTE_CODE,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def merge_lora():
    device = get_device(cuda_id=CUDA_ID)

    base_model = load_base_model()

    print(f"Loading LoRA adapter: {LORA_PATH}")

    lora_model = PeftModel.from_pretrained(
        base_model,
        LORA_PATH,
    )

    print("\n=== Before Merge ===")
    print_model_info(lora_model)
    print_lora_status(lora_model)

    lora_model.to(device)

    print("\nMerging LoRA adapter into base model...")

    merged_model = lora_model.merge_and_unload()

    merged_model.to(device)
    merged_model.eval()

    print("\n=== After Merge ===")
    print_model_info(merged_model)
    print_lora_status(merged_model)

    tokenizer = load_tokenizer()

    print(f"\nSaving merged model to: {MERGED_MODEL_PATH}")

    merged_model.save_pretrained(MERGED_MODEL_PATH)
    tokenizer.save_pretrained(MERGED_MODEL_PATH)

    print("\nMerge completed.")


def main() -> None:
    print_device_info(cuda_id=CUDA_ID)

    merge_lora()


if __name__ == "__main__":
    main()
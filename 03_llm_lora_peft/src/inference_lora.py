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


LORA_PATH = f"{OUTPUT_DIR}/final_lora"

MAX_NEW_TOKENS = 64
TEMPERATURE = 0.1
TOP_P = 0.9
TOP_K = 50
REPETITION_PENALTY = 1.2


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


def build_prompt(instruction: str) -> str:
    return (
        "### Instruction:\n"
        f"{instruction}\n\n"
        "### Response:\n"
    )


def generate_response(instruction: str) -> str:
    tokenizer, model = load_lora_model()
    device = get_device(cuda_id=CUDA_ID)

    prompt = build_prompt(instruction)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            repetition_penalty=REPETITION_PENALTY,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_ids = output_ids[0][inputs["input_ids"].shape[-1]:]

    response = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    )

    return response.strip()


def main() -> None:
    print_device_info(cuda_id=CUDA_ID)

    instruction = "猫は卵を産みますか？"

    print("\n=== Instruction ===")
    print(instruction)

    response = generate_response(instruction)

    print("\n=== Response ===")
    print(response)


if __name__ == "__main__":
    main()
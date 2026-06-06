import torch

from config import (
    CUDA_ID,
    MAX_NEW_TOKENS,
    TEMPERATURE,
    TOP_P,
    TOP_K,
    REPETITION_PENALTY,
)

from device import get_device, print_device_info
from load_model import load_llm


def generate_text(prompt: str) -> str:
    tokenizer, model = load_llm()
    device = get_device(cuda_id=CUDA_ID)

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
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            top_k=TOP_K,
            repetition_penalty=REPETITION_PENALTY,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = output_ids[0][inputs["input_ids"].shape[-1]:]

    generated_text = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    )

    return generated_text


def main() -> None:
    print_device_info(cuda_id=CUDA_ID)

    prompt = "Pythonでリスト内包表記を使うメリットを簡潔に説明してください。"

    print("\n=== Prompt ===")
    print(prompt)

    generated_text = generate_text(prompt)

    print("\n=== Generated Text ===")
    print(generated_text)


if __name__ == "__main__":
    main()
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


def build_messages(user_prompt: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": "あなたは親切で簡潔に回答するAIアシスタントです。",
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]


def generate_chat_response(user_prompt: str) -> str:
    tokenizer, model = load_llm()
    device = get_device(cuda_id=CUDA_ID)

    messages = build_messages(user_prompt)

    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        prompt_text,
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

    response = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    )

    return response.strip()


def main() -> None:
    print_device_info(cuda_id=CUDA_ID)

    user_prompt = "TransformerのSelf-Attentionを初心者向けに簡潔に説明してください。"

    print("\n=== User Prompt ===")
    print(user_prompt)

    response = generate_chat_response(user_prompt)

    print("\n=== Assistant Response ===")
    print(response)


if __name__ == "__main__":
    main()
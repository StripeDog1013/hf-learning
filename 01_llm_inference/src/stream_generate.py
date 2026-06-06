import threading

import torch
from transformers import TextIteratorStreamer

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


def stream_chat_response(user_prompt: str) -> None:
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

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )

    generation_kwargs = {
        **inputs,
        "streamer": streamer,
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": True,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "top_k": TOP_K,
        "repetition_penalty": REPETITION_PENALTY,
        "pad_token_id": tokenizer.eos_token_id,
    }

    thread = threading.Thread(
        target=model.generate,
        kwargs=generation_kwargs,
    )

    thread.start()

    for text in streamer:
        print(text, end="", flush=True)

    thread.join()
    print()


def main() -> None:
    print_device_info(cuda_id=CUDA_ID)

    user_prompt = "PyTorchでGPUを使うメリットを初心者向けに簡潔に説明してください。"

    print("\n=== User Prompt ===")
    print(user_prompt)

    print("\n=== Assistant Response ===")
    stream_chat_response(user_prompt)


if __name__ == "__main__":
    main()
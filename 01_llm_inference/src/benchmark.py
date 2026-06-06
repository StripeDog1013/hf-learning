import time
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


def sync_device(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def get_vram_usage(device: str) -> tuple[float, float]:
    if not device.startswith("cuda"):
        return 0.0, 0.0

    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3

    return allocated, reserved


def benchmark_generate(user_prompt: str) -> None:
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

    input_token_count = inputs["input_ids"].shape[-1]

    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    sync_device(device)
    start_time = time.perf_counter()

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

    sync_device(device)
    end_time = time.perf_counter()

    output_token_count = output_ids.shape[-1]
    generated_token_count = output_token_count - input_token_count

    elapsed_sec = end_time - start_time
    tokens_per_sec = generated_token_count / elapsed_sec

    generated_ids = output_ids[0][input_token_count:]

    generated_text = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    )

    allocated_gb, reserved_gb = get_vram_usage(device)

    print("\n=== Benchmark Result ===")
    print(f"Device              : {device}")
    print(f"Input tokens        : {input_token_count}")
    print(f"Generated tokens    : {generated_token_count}")
    print(f"Elapsed time        : {elapsed_sec:.3f} sec")
    print(f"Tokens/sec          : {tokens_per_sec:.2f}")

    if device.startswith("cuda"):
        peak_allocated = torch.cuda.max_memory_allocated() / 1024**3
        print(f"VRAM allocated      : {allocated_gb:.2f} GB")
        print(f"VRAM reserved       : {reserved_gb:.2f} GB")
        print(f"VRAM peak allocated : {peak_allocated:.2f} GB")

    print("\n=== Generated Text ===")
    print(generated_text.strip())


def main() -> None:
    print_device_info(cuda_id=CUDA_ID)

    user_prompt = (
        "Pythonで機械学習を学ぶときに、"
        "NumPy、PyTorch、Transformersをどの順番で学ぶべきか簡潔に説明してください。"
    )

    print("\n=== Prompt ===")
    print(user_prompt)

    benchmark_generate(user_prompt)


if __name__ == "__main__":
    main()
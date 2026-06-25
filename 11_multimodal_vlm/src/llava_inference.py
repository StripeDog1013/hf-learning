import os

from config import (
    CUDA_ID,
    DEFAULT_IMAGE,
    MAX_IMAGE_SIZE,
    DO_SAMPLE,
    MAX_NEW_TOKENS,
    MODEL_NAME,
    PHYSICAL_CUDA_ID,
    SEED,
    TEMPERATURE,
    TOP_P,
    USE_CUDA_VISIBLE_DEVICES,
    create_dirs,
)

if USE_CUDA_VISIBLE_DEVICES:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(
        PHYSICAL_CUDA_ID
    )

# Mac側のOpenMP競合回避用
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from transformers import (
    LlavaForConditionalGeneration,
    AutoProcessor,
)

from device import (
    clear_cuda_cache,
    get_torch_device,
    print_device_info,
    set_seed,
)
from image_utils import (
    load_image,
    print_image_info,
    resize_keep_aspect,
)
from utils import (
    print_header,
    save_chat_log,
)


def load_llava_model(
    device: torch.device,
):

    processor = AutoProcessor.from_pretrained(
        MODEL_NAME
    )

    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=(
            torch.float16
            if device.type == "cuda"
            else torch.float32
        ),
        low_cpu_mem_usage=True,
    )

    model.to(device)
    model.eval()

    return processor, model


def build_prompt(
    question: str,
) -> str:

    prompt = (
        "USER: <image>\n"
        f"{question}\n"
        "ASSISTANT:"
    )

    return prompt


@torch.no_grad()
def generate_answer(
    image,
    question: str,
    processor,
    model,
    device: torch.device,
) -> str:

    prompt = build_prompt(
        question
    )

    inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    generate_kwargs = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": DO_SAMPLE,
        "pad_token_id": (
            processor.tokenizer.eos_token_id
        ),
    }

    if DO_SAMPLE:
        generate_kwargs["temperature"] = TEMPERATURE
        generate_kwargs["top_p"] = TOP_P

    outputs = model.generate(
        **inputs,
        **generate_kwargs,
    )

    generated_tokens = outputs[
        0,
        inputs["input_ids"].shape[1]:,
    ]

    answer = processor.tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    )

    return answer.strip()


def main():

    print_header("LLaVA Inference")

    create_dirs()
    set_seed(SEED)

    device = get_torch_device(
        cuda_id=CUDA_ID
    )

    print_device_info(
        cuda_id=CUDA_ID
    )

    if not DEFAULT_IMAGE.exists():
        raise FileNotFoundError(
            f"Image not found: {DEFAULT_IMAGE}\n"
            "Please place an image at "
            "data/images/sample.jpg"
        )

    image = load_image(
        DEFAULT_IMAGE
    )

    print_header("Original Image")
    print_image_info(image)
    
    image = resize_keep_aspect(
        image,
        MAX_IMAGE_SIZE,
    )
    
    print_header("Resized Image")
    print_image_info(image)

    processor, model = load_llava_model(
        device=device
    )

    messages = []

    while True:

        question = input(
            "\n画像について質問してください "
            "(終了: q): "
        ).strip()

        if question.lower() in [
            "q",
            "quit",
            "exit",
        ]:
            print("\n終了します。")
            break

        if not question:
            continue

        answer = generate_answer(
            image=image,
            question=question,
            processor=processor,
            model=model,
            device=device,
        )

        print_header("Answer")
        print(answer)

        messages.append(
            {
                "image": str(DEFAULT_IMAGE),
                "question": question,
                "answer": answer,
            }
        )

        clear_cuda_cache()

    log_path = save_chat_log(
        messages,
        prefix="llava_inference",
    )

    print(f"\nSaved chat log: {log_path}")


if __name__ == "__main__":
    main()
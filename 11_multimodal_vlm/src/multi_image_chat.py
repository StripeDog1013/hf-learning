import os
from pathlib import Path

from config import (
    CUDA_ID,
    DO_SAMPLE,
    IMAGE_DIR,
    IMAGE_EXTENSIONS,
    MAX_IMAGE_SIZE,
    MAX_NEW_TOKENS,
    PHYSICAL_CUDA_ID,
    SEED,
    TEMPERATURE,
    TOP_P,
    USE_CUDA_VISIBLE_DEVICES,
    create_dirs,
)

QWEN_VL_MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"

if USE_CUDA_VISIBLE_DEVICES:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(
        PHYSICAL_CUDA_ID
    )

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
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


def list_image_files(
    image_dir: Path,
) -> list[Path]:

    image_files = []

    for file_path in sorted(image_dir.iterdir()):

        if not file_path.is_file():
            continue

        if file_path.suffix.lower() in IMAGE_EXTENSIONS:
            image_files.append(file_path)

    return image_files


def select_images() -> list[Path]:

    image_files = list_image_files(
        IMAGE_DIR
    )

    if len(image_files) == 0:
        raise FileNotFoundError(
            f"No images found in {IMAGE_DIR}"
        )

    print_header("Available Images")

    for i, file_path in enumerate(image_files):
        print(f"[{i}] {file_path.name}")

    raw = input(
        "\n使用する画像番号をカンマ区切りで入力してください "
        "例: 0,1 : "
    ).strip()

    indices = [
        int(x.strip())
        for x in raw.split(",")
        if x.strip()
    ]

    selected = []

    for idx in indices:

        if idx < 0 or idx >= len(image_files):
            raise ValueError(
                f"Invalid image index: {idx}"
            )

        selected.append(
            image_files[idx]
        )

    if len(selected) == 0:
        raise ValueError(
            "No images selected."
        )

    return selected


def load_selected_images(
    image_paths: list[Path],
):

    images = []

    for path in image_paths:

        image = load_image(path)

        print_header(f"Original Image: {path.name}")
        print_image_info(image)

        image = resize_keep_aspect(
            image,
            MAX_IMAGE_SIZE,
        )

        print_header(f"Resized Image: {path.name}")
        print_image_info(image)

        images.append(image)

    return images


def load_qwen_vl_model(
    device: torch.device,
):

    processor = AutoProcessor.from_pretrained(
        QWEN_VL_MODEL_NAME
    )

    model = (
        Qwen2_5_VLForConditionalGeneration
        .from_pretrained(
            QWEN_VL_MODEL_NAME,
            torch_dtype=(
                torch.float16
                if device.type == "cuda"
                else torch.float32
            ),
            device_map=None,
        )
    )

    model.to(device)
    model.eval()

    return processor, model


def build_messages(
    question: str,
    image_count: int,
):

    content = []

    for _ in range(image_count):
        content.append(
            {
                "type": "image",
            }
        )

    content.append(
        {
            "type": "text",
            "text": question,
        }
    )

    messages = [
        {
            "role": "user",
            "content": content,
        }
    ]

    return messages


@torch.no_grad()
def generate_answer(
    images,
    question: str,
    processor,
    model,
    device: torch.device,
) -> str:

    messages = build_messages(
        question=question,
        image_count=len(images),
    )

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=[text],
        images=images,
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

    print_header("Multi Image Chat")

    create_dirs()
    set_seed(SEED)

    device = get_torch_device(
        cuda_id=CUDA_ID
    )

    print_device_info(
        cuda_id=CUDA_ID
    )

    image_paths = select_images()

    images = load_selected_images(
        image_paths
    )

    processor, model = load_qwen_vl_model(
        device=device
    )

    logs = []

    while True:

        question = input(
            "\n複数画像について質問してください "
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
            images=images,
            question=question,
            processor=processor,
            model=model,
            device=device,
        )

        print_header("Answer")
        print(answer)

        logs.append(
            {
                "model": QWEN_VL_MODEL_NAME,
                "images": [
                    str(path)
                    for path in image_paths
                ],
                "question": question,
                "answer": answer,
            }
        )

        clear_cuda_cache()

    log_path = save_chat_log(
        logs,
        prefix="multi_image_chat",
    )

    print(f"\nSaved chat log: {log_path}")


if __name__ == "__main__":
    main()
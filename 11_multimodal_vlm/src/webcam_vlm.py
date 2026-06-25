import os
from pathlib import Path

from config import (
    CUDA_ID,
    DO_SAMPLE,
    MAX_IMAGE_SIZE,
    MAX_NEW_TOKENS,
    OUTPUT_DIR,
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

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import torch
from PIL import Image
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
    print_image_info,
    resize_keep_aspect,
    save_image,
)
from utils import (
    get_timestamp,
    print_header,
    save_chat_log,
)


QWEN_VL_MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"


def capture_frame(
    camera_id: int = 0,
) -> Image.Image:

    cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera: {camera_id}"
        )

    print(
        "\nカメラ映像を表示します。"
        "スペースキーで撮影、qで終了。"
    )

    captured_frame = None

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        cv2.imshow(
            "Webcam - Space: capture / q: quit",
            frame,
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord(" "):
            captured_frame = frame
            break

        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    if captured_frame is None:
        raise RuntimeError(
            "No frame captured."
        )

    rgb_frame = cv2.cvtColor(
        captured_frame,
        cv2.COLOR_BGR2RGB,
    )

    return Image.fromarray(rgb_frame)


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
):
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                },
                {
                    "type": "text",
                    "text": question,
                },
            ],
        }
    ]


@torch.no_grad()
def generate_answer(
    image: Image.Image,
    question: str,
    processor,
    model,
    device: torch.device,
) -> str:

    messages = build_messages(
        question
    )

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=[text],
        images=[image],
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

    print_header("Webcam VLM")

    create_dirs()
    set_seed(SEED)

    device = get_torch_device(
        cuda_id=CUDA_ID
    )

    print_device_info(
        cuda_id=CUDA_ID
    )

    processor, model = load_qwen_vl_model(
        device=device
    )

    logs = []

    while True:

        command = input(
            "\n撮影しますか？ "
            "(Enter: 撮影 / q: 終了): "
        ).strip()

        if command.lower() in [
            "q",
            "quit",
            "exit",
        ]:
            print("\n終了します。")
            break

        image = capture_frame(
            camera_id=0
        )

        print_header("Captured Image")
        print_image_info(image)

        image = resize_keep_aspect(
            image,
            MAX_IMAGE_SIZE,
        )

        print_header("Resized Image")
        print_image_info(image)

        image_path = (
            OUTPUT_DIR
            / f"webcam_{get_timestamp()}.jpg"
        )

        save_image(
            image,
            image_path,
        )

        question = input(
            "\n画像について質問してください: "
        ).strip()

        if not question:
            question = (
                "この画像に写っているものを"
                "日本語で簡潔に説明してください。"
            )

        answer = generate_answer(
            image=image,
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
                "image": str(image_path),
                "question": question,
                "answer": answer,
            }
        )

        clear_cuda_cache()

    log_path = save_chat_log(
        logs,
        prefix="webcam_vlm",
    )

    print(f"\nSaved chat log: {log_path}")


if __name__ == "__main__":
    main()
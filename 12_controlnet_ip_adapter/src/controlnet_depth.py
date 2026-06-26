import os

from config import (
    BASE_MODEL_NAME,
    CONTROLNET_DEPTH_MODEL,
    CUDA_ID,
    GUIDANCE_SCALE,
    HEIGHT,
    IMAGE_DIR,
    MAX_IMAGE_SIZE,
    NEGATIVE_PROMPT,
    NUM_INFERENCE_STEPS,
    OUTPUT_DIR,
    PHYSICAL_CUDA_ID,
    PROMPT,
    SEED,
    USE_CUDA_VISIBLE_DEVICES,
    WIDTH,
    create_dirs,
)

if USE_CUDA_VISIBLE_DEVICES:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(PHYSICAL_CUDA_ID)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np

from diffusers import (
    ControlNetModel,
    StableDiffusionXLControlNetPipeline,
)
from PIL import Image
# from transformers import pipeline
import torch.nn.functional as F
from transformers import (
    AutoImageProcessor,
    AutoModelForDepthEstimation,
)

from device import (
    clear_cuda_cache,
    get_torch_device,
    print_device_info,
)
from image_utils import (
    load_image,
    print_image_info,
    resize_keep_aspect,
    save_image,
)
from utils import (
    get_timestamp,
    print_header,
    save_generation_log,
    save_image_prompt,
    set_seed,
)


DEPTH_MODEL_NAME = "depth-anything/Depth-Anything-V2-Small-hf"


def get_input_image_path():
    image_path = IMAGE_DIR / "sample.jpg"

    if not image_path.exists():
        raise FileNotFoundError(
            f"Input image not found: {image_path}\n"
            "Please place an image at data/images/sample.jpg"
        )

    return image_path


def create_depth_image(
    image: Image.Image,
    device: torch.device,
) -> Image.Image:

    # Depth推定はCPUで固定する
    depth_device = torch.device("cpu")

    processor = AutoImageProcessor.from_pretrained(
        DEPTH_MODEL_NAME
    )

    model = AutoModelForDepthEstimation.from_pretrained(
        DEPTH_MODEL_NAME
    )

    model.to(depth_device)
    model.eval()

    inputs = processor(
        images=image,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(depth_device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        outputs = model(**inputs)

    predicted_depth = outputs.predicted_depth

    prediction = F.interpolate(
        predicted_depth.unsqueeze(1),
        size=(HEIGHT, WIDTH),
        mode="bicubic",
        align_corners=False,
    )

    depth = prediction.squeeze().cpu().numpy()

    depth = (
        (depth - depth.min())
        / (depth.max() - depth.min())
        * 255.0
    )

    depth = depth.astype("uint8")

    depth_image = Image.fromarray(depth)
    depth_image = depth_image.convert("RGB")

    return depth_image


def load_controlnet(
    device: torch.device,
):
    controlnet = ControlNetModel.from_pretrained(
        CONTROLNET_DEPTH_MODEL,
        torch_dtype=(
            torch.float16
            if device.type == "cuda"
            else torch.float32
        ),
        variant=(
            "fp16"
            if device.type == "cuda"
            else None
        ),
        use_safetensors=True,
    )

    controlnet.to(device)

    return controlnet


def load_pipeline(
    device: torch.device,
    controlnet,
):
    dtype = (
        torch.float16
        if device.type == "cuda"
        else torch.float32
    )

    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        BASE_MODEL_NAME,
        controlnet=controlnet,
        torch_dtype=dtype,
        variant=(
            "fp16"
            if device.type == "cuda"
            else None
        ),
        use_safetensors=True,
    )

    pipe.to(device)

    # 念のため、主要コンポーネントを明示的に移動
    pipe.text_encoder.to(device)
    pipe.text_encoder_2.to(device)
    pipe.unet.to(device)
    pipe.vae.to(device)
    pipe.controlnet.to(device)

    try:
        pipe.enable_xformers_memory_efficient_attention()
        print_header("xFormers enabled.")
    except Exception as e:
        print_header(f"xFormers not enabled: {e}")

    return pipe


def prepare_control_image(
    device: torch.device,
):
    image_path = get_input_image_path()

    image = load_image(image_path)

    print_header("Original Image")
    print_image_info(image)

    image = resize_keep_aspect(
        image,
        MAX_IMAGE_SIZE,
    )

    image = image.resize(
        (WIDTH, HEIGHT)
    )

    print_header("Resized Image")
    print_image_info(image)

    depth_image = create_depth_image(
        image=image,
        device=device,
    )

    print_header("Depth Image")
    print_image_info(depth_image)

    return image_path, image, depth_image


def generate_image(
    pipe,
    control_image,
    device: torch.device,
):
    generator = torch.Generator(
        device=device
    ).manual_seed(SEED)

    print("text_encoder:", next(pipe.text_encoder.parameters()).device)
    print("text_encoder_2:", next(pipe.text_encoder_2.parameters()).device)
    print("unet:", next(pipe.unet.parameters()).device)
    print("vae:", next(pipe.vae.parameters()).device)
    print("controlnet:", next(pipe.controlnet.parameters()).device)

    image = pipe(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        image=control_image,
        height=HEIGHT,
        width=WIDTH,
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        controlnet_conditioning_scale=0.8,
        generator=generator,
    ).images[0]

    return image


def main():
    print_header("ControlNet Depth")

    create_dirs()
    set_seed(SEED)

    device = get_torch_device(
        cuda_id=CUDA_ID
    )

    print_device_info(
        cuda_id=CUDA_ID
    )

    input_path, original_image, depth_image = prepare_control_image(
        device=device
    )

    timestamp = get_timestamp()

    depth_path = (
        OUTPUT_DIR
        / f"controlnet_depth_input_{timestamp}.png"
    )

    save_image(
        depth_image,
        depth_path,
    )

    controlnet = load_controlnet(
        device=device
    )

    pipe = load_pipeline(
        device=device,
        controlnet=controlnet,
    )

    generated_image = generate_image(
        pipe=pipe,
        control_image=depth_image,
        device=device,
    )

    output_path = (
        OUTPUT_DIR
        / f"controlnet_depth_{timestamp}.png"
    )

    save_image(
        generated_image,
        output_path,
    )

    save_image_prompt(
        output_path,
        PROMPT,
    )

    log_path = save_generation_log(
        {
            "script": "controlnet_depth.py",
            "base_model": BASE_MODEL_NAME,
            "controlnet_model": CONTROLNET_DEPTH_MODEL,
            "depth_model": DEPTH_MODEL_NAME,
            "input_image": str(input_path),
            "depth_image": str(depth_path),
            "output_image": str(output_path),
            "prompt": PROMPT,
            "negative_prompt": NEGATIVE_PROMPT,
            "height": HEIGHT,
            "width": WIDTH,
            "num_inference_steps": NUM_INFERENCE_STEPS,
            "guidance_scale": GUIDANCE_SCALE,
            "controlnet_conditioning_scale": 0.8,
            "seed": SEED,
        }
    )

    print_header("Saved")
    print(f"Depth : {depth_path}")
    print(f"Image : {output_path}")
    print(f"Log   : {log_path}")

    clear_cuda_cache()


if __name__ == "__main__":
    main()
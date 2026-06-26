import os

from config import (
    BASE_MODEL_NAME,
    CONTROLNET_OPENPOSE_MODEL,
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
    os.environ["CUDA_VISIBLE_DEVICES"] = str(
        PHYSICAL_CUDA_ID
    )

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from controlnet_aux import OpenposeDetector
from diffusers import (
    ControlNetModel,
    StableDiffusionXLControlNetPipeline,
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


OPENPOSE_PREPROCESSOR_MODEL = (
    "lllyasviel/ControlNet"
)


def get_input_image_path():
    image_path = IMAGE_DIR / "sample.jpg"

    if not image_path.exists():
        raise FileNotFoundError(
            f"Input image not found: {image_path}\n"
            "Please place an image at data/images/sample.jpg"
        )

    return image_path


def create_openpose_image(
    image,
):

    detector = OpenposeDetector.from_pretrained(
        OPENPOSE_PREPROCESSOR_MODEL
    )

    pose_image = detector(
        image,
        include_body=True,
        include_hand=False,
        include_face=False,
    )

    pose_image = pose_image.resize(
        (WIDTH, HEIGHT)
    )

    return pose_image


def load_controlnet(
    device: torch.device,
):
    controlnet = ControlNetModel.from_pretrained(
        CONTROLNET_OPENPOSE_MODEL,
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

    # Depth版と同様、主要コンポーネントを明示的に同じdeviceへ揃える
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


def prepare_control_image():
    image_path = get_input_image_path()

    image = load_image(
        image_path
    )

    print_header("Original Image")
    print_image_info(
        image
    )

    image = resize_keep_aspect(
        image,
        MAX_IMAGE_SIZE,
    )

    image = image.resize(
        (WIDTH, HEIGHT)
    )

    print_header("Resized Image")
    print_image_info(
        image
    )

    pose_image = create_openpose_image(
        image
    )

    print_header("OpenPose Image")
    print_image_info(
        pose_image
    )

    return image_path, image, pose_image


def generate_image(
    pipe,
    control_image,
    device: torch.device,
):
    generator = torch.Generator(
        device=device
    ).manual_seed(SEED)

    image = pipe(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        image=control_image,
        height=HEIGHT,
        width=WIDTH,
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        controlnet_conditioning_scale=0.9,
        generator=generator,
    ).images[0]

    return image


def main():

    print_header("ControlNet OpenPose")

    create_dirs()
    set_seed(SEED)

    device = get_torch_device(
        cuda_id=CUDA_ID
    )

    print_device_info(
        cuda_id=CUDA_ID
    )

    input_path, original_image, pose_image = (
        prepare_control_image()
    )

    timestamp = get_timestamp()

    pose_path = (
        OUTPUT_DIR
        / f"controlnet_openpose_input_{timestamp}.png"
    )

    save_image(
        pose_image,
        pose_path,
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
        control_image=pose_image,
        device=device,
    )

    output_path = (
        OUTPUT_DIR
        / f"controlnet_openpose_{timestamp}.png"
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
            "script": "controlnet_openpose.py",
            "base_model": BASE_MODEL_NAME,
            "controlnet_model": CONTROLNET_OPENPOSE_MODEL,
            "openpose_preprocessor_model": OPENPOSE_PREPROCESSOR_MODEL,
            "input_image": str(input_path),
            "pose_image": str(pose_path),
            "output_image": str(output_path),
            "prompt": PROMPT,
            "negative_prompt": NEGATIVE_PROMPT,
            "height": HEIGHT,
            "width": WIDTH,
            "num_inference_steps": NUM_INFERENCE_STEPS,
            "guidance_scale": GUIDANCE_SCALE,
            "controlnet_conditioning_scale": 0.9,
            "seed": SEED,
        }
    )

    print_header("Saved")
    print(f"Pose  : {pose_path}")
    print(f"Image : {output_path}")
    print(f"Log   : {log_path}")

    clear_cuda_cache()


if __name__ == "__main__":
    main()
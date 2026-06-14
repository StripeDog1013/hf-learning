from pathlib import Path


# ==================================================
# Model
# ==================================================

# MODEL_NAME = "stabilityai/sd-turbo"
# 通常のStable Diffusion v1.5を使う場合はこちら
MODEL_NAME = "runwayml/stable-diffusion-v1-5"

param_dict = {
    "stabilityai/sd-turbo":[4, 0.0],
    "runwayml/stable-diffusion-v1-5":[50, 7.5]
    }

# ==================================================
# Device
# ==================================================

CUDA_ID = 0

USE_CUDA_VISIBLE_DEVICES = True

PHYSICAL_CUDA_ID = 0


# ==================================================
# Generation
# ==================================================

PROMPT = (
    "a cute shiba inu wearing a space suit, "
    "cinematic lighting, highly detailed"
)

NEGATIVE_PROMPT = (
    "low quality, blurry, distorted"
)

NUM_INFERENCE_STEPS, GUIDANCE_SCALE = param_dict[MODEL_NAME]

WIDTH = 512

HEIGHT = 512

SEED = 42


# ==================================================
# Output
# ==================================================

OUTPUT_DIR = Path("outputs")

OUTPUT_FILE = "generated.png"
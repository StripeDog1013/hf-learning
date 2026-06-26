from pathlib import Path


# ============================================================
# Project Paths
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data"

IMAGE_DIR = DATA_DIR / "images"
CONTROL_DIR = DATA_DIR / "control"
REFERENCE_DIR = DATA_DIR / "reference"

MODEL_DIR = ROOT_DIR / "models"
OUTPUT_DIR = ROOT_DIR / "outputs"
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"


# ============================================================
# GPU
# ============================================================

# RTX4070Ti : 0
# RTX4000 Ada : 1
PHYSICAL_CUDA_ID = 0

USE_CUDA_VISIBLE_DEVICES = True

CUDA_ID = 0


# ============================================================
# Base Model
# ============================================================

BASE_MODEL_NAME = (
    "stabilityai/stable-diffusion-xl-base-1.0"
)


# ============================================================
# ControlNet
# ============================================================

CONTROLNET_CANNY_MODEL = (
    "diffusers/controlnet-canny-sdxl-1.0"
)

CONTROLNET_DEPTH_MODEL = (
    "diffusers/controlnet-depth-sdxl-1.0"
)

CONTROLNET_OPENPOSE_MODEL = (
    "r3gm/controlnet-openpose-sdxl-1.0-fp16"
)


# ============================================================
# IP-Adapter
# ============================================================

IP_ADAPTER_REPO = "h94/IP-Adapter"

IP_ADAPTER_WEIGHT = (
    "ip-adapter_sdxl.bin"
)


# ============================================================
# Generation
# ============================================================

PROMPT = (
    "A cute Shiba Inu, high quality,"
    " realistic, 8k"
)

NEGATIVE_PROMPT = (
    "low quality, blurry,"
    " bad anatomy"
)

HEIGHT = 1024

WIDTH = 1024

NUM_INFERENCE_STEPS = 30

GUIDANCE_SCALE = 7.5

SEED = 42


# ============================================================
# Image
# ============================================================

IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".JPG",
    ".png",
    ".bmp",
    ".webp",
)

MAX_IMAGE_SIZE = 1024


# ============================================================
# Utility
# ============================================================

def create_dirs():

    directories = [

        DATA_DIR,

        IMAGE_DIR,

        CONTROL_DIR,

        REFERENCE_DIR,

        MODEL_DIR,

        OUTPUT_DIR,

        CHECKPOINT_DIR,

    ]

    for directory in directories:

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )
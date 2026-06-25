from pathlib import Path


# ============================================================
# Project Paths
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data"
IMAGE_DIR = DATA_DIR / "images"
SAMPLE_DIR = DATA_DIR / "samples"

MODEL_DIR = ROOT_DIR / "models"
OUTPUT_DIR = ROOT_DIR / "outputs"
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"


# ============================================================
# GPU Settings
# ============================================================

# RTX4070Ti : 0
# RTX4000 Ada : 1
PHYSICAL_CUDA_ID = 0

USE_CUDA_VISIBLE_DEVICES = True

CUDA_ID = 0


# ============================================================
# Model Settings
# ============================================================

# 最初はLLaVAはidx=0
# Qwen-VLを試す場合はidx=1
dict_model_name = {
    0:"llava-hf/llava-1.5-7b-hf",
    1:"Qwen/Qwen2.5-VL-3B-Instruct",
    }
MODEL_NAME = dict_model_name[0]

# ============================================================
# Generation
# ============================================================

MAX_NEW_TOKENS = 256

DO_SAMPLE = False

TEMPERATURE = 0.0

TOP_P = 1.0


# ============================================================
# Image
# ============================================================

MAX_IMAGE_SIZE = 672

IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".JPG",
    ".png",
    ".bmp",
    ".webp",
)

DEFAULT_IMAGE = IMAGE_DIR / "sample.jpg"


# ============================================================
# Random Seed
# ============================================================

SEED = 42


# ============================================================
# Utility
# ============================================================

def create_dirs():

    for directory in [
        DATA_DIR,
        IMAGE_DIR,
        SAMPLE_DIR,
        MODEL_DIR,
        OUTPUT_DIR,
        CHECKPOINT_DIR,
    ]:

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )
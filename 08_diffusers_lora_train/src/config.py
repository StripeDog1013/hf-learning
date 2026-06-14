from pathlib import Path


# ==================================================
# Model
# ==================================================

MODEL_NAME = "runwayml/stable-diffusion-v1-5"

# 軽量確認用に使う場合
# MODEL_NAME = "stabilityai/sd-turbo"

param_dict = {
    "stabilityai/sd-turbo":[4, 0.0],
    "runwayml/stable-diffusion-v1-5":[50, 7.5]
    }

# ==================================================
# Dataset
# ==================================================

DATA_DIR = Path("data/daifuku")

IMAGE_EXTENSIONS = [
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
]

INSTANCE_PROMPT = (
    "a photo of daifuku_cat" # "a photo of sks cat"
)

CLASS_PROMPT = (
    "a photo of a cat"
)


# ==================================================
# Image
# ==================================================

IMAGE_SIZE = 512


# ==================================================
# Training
# ==================================================

NUM_EPOCHS = 30

BATCH_SIZE = 1

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 0.01

GRADIENT_ACCUMULATION_STEPS = 4

MAX_GRAD_NORM = 1.0

SEED = 42


# ==================================================
# LoRA
# ==================================================

LORA_RANK = 4

LORA_ALPHA = 4

LORA_DROPOUT = 0.0


# ==================================================
# Device
# ==================================================

CUDA_ID = 0

USE_CUDA_VISIBLE_DEVICES = True

PHYSICAL_CUDA_ID = 0


# ==================================================
# Output
# ==================================================

OUTPUT_DIR = Path("checkpoints")

LORA_OUTPUT_DIR = OUTPUT_DIR / "lora"

LOG_DIR = Path("logs")

SAMPLE_OUTPUT_DIR = Path("outputs")


# ==================================================
# Inference
# ==================================================

INFERENCE_PROMPT = (
    # "a photo of sks cat wearing a wizard hat, "
    # "a photo of daifuku_cat wearing a wizard hat,"
    # "a photo of daifuku_cat, blue eyes, cream white fur, gray tabby markings on face, striped tail, highly detailed, realistic"
    "a photo of daifuku_cat wearing a wizard hat, blue eyes, cream white fur, gray tabby markings on face, striped tail, realistic, highly detailed, soft lighting,"
    # professional daifuku_cat photography of daifuku_cat, blue eyes, cream colored fur, gray facial stripes, striped tail, natural lighting, ultra realistic
    # cute portrait of daifuku_cat, bright blue eyes, cream white fur, gray striped face, fluffy tail, cinematic lighting
    "high quality, detailed"
)

NEGATIVE_PROMPT = (
    # "low quality, blurry, distorted"
    "cartoon, anime, painting, illustration, blurry, low quality, distorted, deformed, extra limbs, extra ears, duplicate, bad anatomy, cropped"
)

NUM_INFERENCE_STEPS, GUIDANCE_SCALE = param_dict[MODEL_NAME]
#"stabilityai/sd-turbo":[4, 0.0],
#"runwayml/stable-diffusion-v1-5":[50, 7.5]

INFERENCE_SEED = 1234
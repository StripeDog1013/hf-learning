from pathlib import Path


# ==================================================
# Dataset
# ==================================================

DATASET_NAME = "rishitdagli/cppe-5"

USE_LOCAL_IMAGE_FOLDER = True

LOCAL_DATA_DIR = (
    Path(__file__).parent.parent
    / "data"
    / "cppe-5"
)


# ==================================================
# Model
# ==================================================

MODEL_NAME = "facebook/detr-resnet-50" # facebook/detr-resnet-101

NUM_LABELS = 5

IMAGE_SIZE = 800


# ==================================================
# Training
# ==================================================

NUM_EPOCHS = 5 # 20 or 50

BATCH_SIZE = 8 # 8

LEARNING_RATE = 1e-5 # 5e-5

WEIGHT_DECAY = 1e-4

WARMUP_RATIO = 0.1

SEED = 42


# ==================================================
# Logging
# ==================================================

LOGGING_STEPS = 10

SAVE_STEPS = 100

EVAL_STEPS = 100

SAVE_TOTAL_LIMIT = 3

LOAD_BEST_MODEL_AT_END = True

EVALUATION_STRATEGY = "steps"

SAVE_STRATEGY = "steps"


# ==================================================
# Output
# ==================================================

OUTPUT_DIR = "checkpoints"

LOG_DIR = "logs"


# ==================================================
# Device
# ==================================================

CUDA_ID = 0

USE_CUDA_VISIBLE_DEVICES = True

PHYSICAL_CUDA_ID = 0
from pathlib import Path


# ============================================================
# Project Paths
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"
OUTPUT_DIR = ROOT_DIR / "outputs"
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"


# ============================================================
# GPU Settings
# ============================================================

# 物理GPU ID
# RTX4070Ti     : 0
# RTX4000 Ada   : 1
PHYSICAL_CUDA_ID = 0

# CUDA_VISIBLE_DEVICES を使って、
# PyTorchから見えるGPUを1枚に制限する
USE_CUDA_VISIBLE_DEVICES = True

# CUDA_VISIBLE_DEVICES使用時は基本的に cuda:0
CUDA_ID = 0


# ============================================================
# LLM Settings
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

MAX_NEW_TOKENS = 128
TEMPERATURE = 0.0
TOP_P = 1.0
DO_SAMPLE = False


# ============================================================
# Agent Settings
# ============================================================

MAX_AGENT_STEPS = 5

TOOL_CALL_START = "<tool_call>"
TOOL_CALL_END = "</tool_call>"

OBSERVATION_START = "<observation>"
OBSERVATION_END = "</observation>"


# ============================================================
# Random Seed
# ============================================================

SEED = 42


# ============================================================
# Utility
# ============================================================

def create_dirs():
    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
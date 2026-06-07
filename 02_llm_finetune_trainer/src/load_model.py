import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

from config import (
    CUDA_ID,
    MODEL_NAME,
    TRUST_REMOTE_CODE,
    USE_FP16,
)

from device import get_device


def load_tokenizer():
    print(f"Loading tokenizer: {MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=TRUST_REMOTE_CODE,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def load_model():
    device = get_device(cuda_id=CUDA_ID)

    print(f"Loading model: {MODEL_NAME}")
    print(f"Device: {device}")

    dtype = torch.float32

    if device.startswith("cuda") and USE_FP16:
        dtype = torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=TRUST_REMOTE_CODE,
        dtype=dtype,
    )

    model.to(device)
    model.train()

    return model


def load_llm():
    tokenizer = load_tokenizer()
    model = load_model()

    return tokenizer, model
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

from config import (
    MODEL_NAME,
    TRUST_REMOTE_CODE,
    USE_FP16,
    CUDA_ID,
)

from device import get_device, print_device_info


def load_tokenizer():
    print(f"Loading tokenizer: {MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=TRUST_REMOTE_CODE,
    )

    return tokenizer


def load_model():
    device = get_device(cuda_id=CUDA_ID)

    print(f"Loading model: {MODEL_NAME}")
    print_device_info(cuda_id=CUDA_ID)

    dtype = torch.float32

    if device.startswith("cuda") and USE_FP16:
        dtype = torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=TRUST_REMOTE_CODE,
        dtype=dtype,
    )

    model.to(device)
    model.eval()

    return model


def load_llm():
    tokenizer = load_tokenizer()
    model = load_model()

    return tokenizer, model
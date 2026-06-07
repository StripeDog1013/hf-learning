import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from config import (
    MODEL_NAME,
    TRUST_REMOTE_CODE,
)


def load_tokenizer():

    print(
        f"Loading tokenizer: "
        f"{MODEL_NAME}"
    )

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            MODEL_NAME,
            trust_remote_code=
            TRUST_REMOTE_CODE,
        )
    )

    if tokenizer.pad_token is None:

        tokenizer.pad_token = (
            tokenizer.eos_token
        )

    return tokenizer


def load_model():

    print(
        f"Loading model: "
        f"{MODEL_NAME}"
    )

    model = (
        AutoModelForCausalLM
        .from_pretrained(
            MODEL_NAME,
            trust_remote_code=
            TRUST_REMOTE_CODE,
            dtype=torch.float32,
        )
    )

    return model


def load_llm():

    tokenizer = load_tokenizer()

    model = load_model()

    return tokenizer, model
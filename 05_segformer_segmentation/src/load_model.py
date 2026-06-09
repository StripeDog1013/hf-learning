from transformers import SegformerForSemanticSegmentation

from config import MODEL_NAME
from device import get_device


def load_model(cuda_id: int = 0):
    device = get_device(cuda_id)

    print(f"Loading model: {MODEL_NAME}")

    model = SegformerForSemanticSegmentation.from_pretrained(
        MODEL_NAME,
    )

    model.to(device)
    model.train()

    return model


def load_model_for_inference(
    model_path: str,
    cuda_id: int = 0,
):
    device = get_device(cuda_id)

    print(f"Loading fine-tuned model: {model_path}")

    model = SegformerForSemanticSegmentation.from_pretrained(
        model_path,
    )

    model.to(device)
    model.eval()

    return model
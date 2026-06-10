from transformers import DetrForObjectDetection

from config import (
    MODEL_NAME,
    NUM_LABELS,
)

from device import get_device


ID2LABEL = {
    0: "coverall",
    1: "face_shield",
    2: "gloves",
    3: "goggles",
    4: "mask",
}

LABEL2ID = {
    label: idx
    for idx, label in ID2LABEL.items()
}


def load_model(cuda_id: int = 0):
    device = get_device(cuda_id)

    print(f"Loading model: {MODEL_NAME}")

    model = DetrForObjectDetection.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
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

    model = DetrForObjectDetection.from_pretrained(
        model_path,
    )

    model.to(device)
    model.eval()

    return model
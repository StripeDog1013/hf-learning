from transformers import ViTForImageClassification

from config import MODEL_NAME
from device import get_device


def get_label_names(train_dataset):
    if "labels" in train_dataset.features:
        return train_dataset.features["labels"].names

    if "label" in train_dataset.features:
        return train_dataset.features["label"].names

    raise ValueError(
        f"Label feature not found: {train_dataset.features}"
    )


def build_label_mappings(train_dataset):
    label_names = get_label_names(train_dataset)

    id2label = {
        idx: label
        for idx, label in enumerate(label_names)
    }

    label2id = {
        label: idx
        for idx, label in id2label.items()
    }

    return id2label, label2id


def load_model(train_dataset, cuda_id: int = 0):
    device = get_device(cuda_id)

    id2label, label2id = build_label_mappings(
        train_dataset
    )

    print(f"Loading model: {MODEL_NAME}")
    print(f"Number of labels: {len(id2label)}")

    model = ViTForImageClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    model.to(device)
    model.train()

    return model


def load_model_for_inference(model_path: str, cuda_id: int = 0):
    device = get_device(cuda_id)

    print(f"Loading fine-tuned model: {model_path}")

    model = ViTForImageClassification.from_pretrained(
        model_path,
    )

    model.to(device)
    model.eval()

    return model
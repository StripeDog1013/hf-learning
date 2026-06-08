from pathlib import Path
from datasets import load_dataset

DATASET_NAME = "AI-Lab-Makerere/beans"
OUTPUT_ROOT = Path("data/images")

dataset = load_dataset(DATASET_NAME)

for split_name in ["train", "validation"]:
    split_dataset = dataset[split_name]

    label_names = split_dataset.features["labels"].names

    for i, sample in enumerate(split_dataset):
        image = sample["image"]
        label_id = sample["labels"]
        label_name = label_names[label_id]

        output_dir = OUTPUT_ROOT / split_name / label_name
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / f"{i:05d}.jpg"

        image.convert("RGB").save(output_path)
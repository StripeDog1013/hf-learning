from collections import Counter

import torch

from dataset import load_datasets
from image_processor import (
    load_image_processor,
    transform_dataset,
)


def check_dataset(dataset, name: str):
    print(f"\n=== Checking {name} ===")

    label_counter = Counter()
    min_label = 999999
    max_label = -1

    for i in range(len(dataset)):
        sample = dataset[i]
        labels = sample["labels"]

        unique_labels = torch.unique(labels)

        for label in unique_labels.tolist():
            label_counter[label] += 1

        min_label = min(min_label, int(labels.min()))
        max_label = max(max_label, int(labels.max()))

        invalid = [
            label
            for label in unique_labels.tolist()
            if label != 255 and not (0 <= label <= 149)
        ]

        if invalid:
            print(f"Invalid labels found at index {i}")
            print(f"Invalid labels: {invalid}")
            return

        if (i + 1) % 1000 == 0:
            print(f"Checked {i + 1} samples")

    print("OK")
    print(f"min label: {min_label}")
    print(f"max label: {max_label}")
    print(f"unique labels: {sorted(label_counter.keys())[:20]} ...")
    print(f"num unique labels: {len(label_counter)}")
    all_labels = sorted(label_counter.keys())
    print(all_labels[-10:])
    
def main():
    train_dataset, valid_dataset = load_datasets()

    image_processor = load_image_processor()

    train_dataset = transform_dataset(
        train_dataset,
        image_processor,
    )

    valid_dataset = transform_dataset(
        valid_dataset,
        image_processor,
    )

    check_dataset(train_dataset, "train")
    check_dataset(valid_dataset, "validation")


if __name__ == "__main__":
    main()
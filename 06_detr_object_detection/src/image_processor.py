import torch

from transformers import AutoImageProcessor

from config import MODEL_NAME


def load_image_processor():
    print(f"Loading image processor: {MODEL_NAME}")

    return AutoImageProcessor.from_pretrained(
        MODEL_NAME,
    )


def convert_objects_to_annotations(example):
    objects = example["objects"]

    annotations = []

    for bbox, category, area in zip(
        objects["bbox"],
        objects["category"],
        objects["area"],
    ):
        annotations.append(
            {
                "bbox": bbox,
                "category_id": category,
                "area": area,
                "iscrowd": 0,
            }
        )

    return {
        "image_id": example["image_id"],
        "annotations": annotations,
    }


def transform_example(example, image_processor):
    image = example["image"].convert("RGB")

    target = convert_objects_to_annotations(
        example,
    )

    encoding = image_processor(
        images=image,
        annotations=target,
        return_tensors="pt",
    )

    return {
        "pixel_values": encoding["pixel_values"][0],
        "labels": encoding["labels"][0],
    }


def collate_fn(batch):
        pixel_values_list = [
            item["pixel_values"]
            for item in batch
        ]

        labels = [
            item["labels"]
            for item in batch
        ]

        max_height = max(
            pixel_values.shape[1]
            for pixel_values in pixel_values_list
        )

        max_width = max(
            pixel_values.shape[2]
            for pixel_values in pixel_values_list
        )

        padded_pixel_values = []
        pixel_masks = []

        for pixel_values in pixel_values_list:
            _, height, width = pixel_values.shape

            padded = torch.zeros(
                3,
                max_height,
                max_width,
                dtype=pixel_values.dtype,
            )

            padded[
                :,
                :height,
                :width,
            ] = pixel_values

            pixel_mask = torch.zeros(
                max_height,
                max_width,
                dtype=torch.long,
            )

            pixel_mask[
                :height,
                :width,
            ] = 1

            padded_pixel_values.append(padded)
            pixel_masks.append(pixel_mask)

        return {
            "pixel_values": torch.stack(padded_pixel_values),
            "pixel_mask": torch.stack(pixel_masks),
            "labels": labels,
        }


class ObjectDetectionDataset:
    def __init__(self, dataset, image_processor):
        self.dataset = dataset
        self.image_processor = image_processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int):
        example = self.dataset[index]

        return transform_example(
            example,
            self.image_processor,
        )


def transform_dataset(dataset, image_processor):
    return ObjectDetectionDataset(
        dataset,
        image_processor,
    )


def print_processed_sample(dataset, index: int = 0):
    sample = dataset[index]

    print("\n=== Processed Sample ===")
    print(f"Keys               : {sample.keys()}")
    print(f"pixel_values shape : {sample['pixel_values'].shape}")
    print(f"labels keys        : {sample['labels'].keys()}")

    for key, value in sample["labels"].items():
        if hasattr(value, "shape"):
            print(f"{key:15}: {value.shape}")
        else:
            print(f"{key:15}: {value}")


def main():
    from dataset import load_datasets

    image_processor = load_image_processor()

    train_dataset, _ = load_datasets()

    processed_dataset = transform_dataset(
        train_dataset,
        image_processor,
    )

    print_processed_sample(
        processed_dataset,
    )


if __name__ == "__main__":
    main()
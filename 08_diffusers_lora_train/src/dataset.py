from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from config import (
    DATA_DIR,
    IMAGE_EXTENSIONS,
    IMAGE_SIZE,
    INSTANCE_PROMPT,
)


class DreamBoothLoRADataset(Dataset):
    def __init__(
        self,
        data_dir: str | Path = DATA_DIR,
        prompt: str = INSTANCE_PROMPT,
        image_size: int = IMAGE_SIZE,
    ):
        self.data_dir = Path(data_dir)
        self.prompt = prompt
        self.image_size = image_size

        self.image_paths = self._collect_images()

        if len(self.image_paths) == 0:
            raise FileNotFoundError(
                f"No images found in: {self.data_dir}"
            )

        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    image_size,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.CenterCrop(
                    image_size,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.5],
                    [0.5],
                ),
            ]
        )

    def _collect_images(self):
        image_paths = []

        for ext in IMAGE_EXTENSIONS:
            image_paths.extend(
                self.data_dir.glob(f"*{ext}")
            )

        return sorted(image_paths)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index: int):
        image_path = self.image_paths[index]

        image = Image.open(image_path).convert("RGB")

        pixel_values = self.transform(image)

        return {
            "pixel_values": pixel_values,
            "prompt": self.prompt,
            "image_path": str(image_path),
        }


def load_dataset():
    return DreamBoothLoRADataset()


def print_dataset_info(dataset):
    print("\n=== Dataset Information ===")
    print(f"Rows : {len(dataset)}")

    sample = dataset[0]

    print("\n=== Sample ===")
    print(f"Keys          : {sample.keys()}")
    print(f"Pixel Shape   : {sample['pixel_values'].shape}")
    print(f"Prompt        : {sample['prompt']}")
    print(f"Image Path    : {sample['image_path']}")


def main():
    dataset = load_dataset()

    print_dataset_info(dataset)


if __name__ == "__main__":
    main()
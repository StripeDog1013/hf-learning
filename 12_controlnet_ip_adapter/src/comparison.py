from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from config import (
    OUTPUT_DIR,
    create_dirs,
)
from image_utils import (
    load_image,
    save_image,
)
from utils import (
    get_timestamp,
    print_header,
)


TARGET_PREFIXES = [
    "sdxl_txt2img_",
    "controlnet_canny_",
    "controlnet_depth_",
    "controlnet_openpose_",
    "ip_adapter_",
]


def find_latest_image(
    prefix: str,
) -> Path | None:

    files = sorted(
        OUTPUT_DIR.glob(f"{prefix}*.png"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if len(files) == 0:
        return None

    return files[0]


def resize_for_grid(
    image: Image.Image,
    size: tuple[int, int],
) -> Image.Image:

    return image.resize(size)


def add_label(
    image: Image.Image,
    label: str,
    label_height: int = 40,
) -> Image.Image:

    width, height = image.size

    canvas = Image.new(
        "RGB",
        (width, height + label_height),
        "white",
    )

    canvas.paste(
        image,
        (0, label_height),
    )

    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype(
            "arial.ttf",
            18,
        )
    except Exception:
        font = ImageFont.load_default()

    draw.text(
        (10, 10),
        label,
        fill="black",
        font=font,
    )

    return canvas


def make_comparison_grid(
    image_items: list[tuple[str, Image.Image]],
    image_size: tuple[int, int] = (384, 384),
) -> Image.Image:

    labeled_images = []

    for label, image in image_items:

        image = resize_for_grid(
            image,
            image_size,
        )

        image = add_label(
            image,
            label,
        )

        labeled_images.append(image)

    total_width = sum(
        image.width
        for image in labeled_images
    )

    max_height = max(
        image.height
        for image in labeled_images
    )

    grid = Image.new(
        "RGB",
        (total_width, max_height),
        "white",
    )

    x = 0

    for image in labeled_images:

        grid.paste(
            image,
            (x, 0),
        )

        x += image.width

    return grid


def main():

    print_header("Comparison")

    create_dirs()

    image_items = []

    for prefix in TARGET_PREFIXES:

        path = find_latest_image(
            prefix
        )

        if path is None:
            print(
                f"Not found: {prefix}*.png"
            )
            continue

        label = prefix.rstrip("_")

        print(
            f"Found {label}: {path.name}"
        )

        image = load_image(
            path
        )

        image_items.append(
            (label, image)
        )

    if len(image_items) == 0:
        raise RuntimeError(
            "No generated images found."
        )

    grid = make_comparison_grid(
        image_items
    )

    output_path = (
        OUTPUT_DIR
        / f"comparison_{get_timestamp()}.png"
    )

    save_image(
        grid,
        output_path,
    )

    print_header("Saved")
    print(output_path)


if __name__ == "__main__":
    main()
from pathlib import Path

from PIL import Image


def load_image(
    image_path: str | Path,
) -> Image.Image:

    image = Image.open(
        image_path
    )

    return image.convert("RGB")


def save_image(
    image: Image.Image,
    save_path: str | Path,
):

    Path(save_path).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    image.save(save_path)


def resize_image(
    image: Image.Image,
    size: tuple[int, int],
) -> Image.Image:

    return image.resize(size)

def resize_keep_aspect(
    image: Image.Image,
    max_size: int,
) -> Image.Image:

    image = image.copy()

    image.thumbnail(
        (max_size, max_size)
    )

    return image

def print_image_info(
    image: Image.Image,
):

    print("\n=== Image Information ===")

    print(
        f"Width  : {image.width}"
    )

    print(
        f"Height : {image.height}"
    )

    print(
        f"Mode   : {image.mode}"
    )


def show_image(
    image: Image.Image,
):

    image.show()


if __name__ == "__main__":

    from config import DEFAULT_IMAGE

    image = load_image(
        DEFAULT_IMAGE
    )

    print_image_info(
        image
    )

    show_image(
        image
    )
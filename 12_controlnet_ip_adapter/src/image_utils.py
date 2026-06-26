from pathlib import Path

import cv2

import numpy as np

from PIL import Image


def load_image(
    image_path,
):

    image = Image.open(
        image_path
    )

    return image.convert("RGB")


def save_image(
    image,
    save_path,
):

    Path(save_path).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    image.save(save_path)


def resize_keep_aspect(
    image,
    max_size,
):

    image = image.copy()

    image.thumbnail(
        (max_size, max_size)
    )

    return image


def pil_to_cv(
    image,
):

    image = np.array(image)

    return cv2.cvtColor(
        image,
        cv2.COLOR_RGB2BGR,
    )


def cv_to_pil(
    image,
):

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB,
    )

    return Image.fromarray(image)


def create_canny_image(
    image,
    low_threshold=100,
    high_threshold=200,
):

    image = pil_to_cv(
        image
    )

    edge = cv2.Canny(
        image,
        low_threshold,
        high_threshold,
    )

    edge = cv2.cvtColor(
        edge,
        cv2.COLOR_GRAY2RGB,
    )

    return cv_to_pil(
        edge
    )


def print_image_info(
    image,
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
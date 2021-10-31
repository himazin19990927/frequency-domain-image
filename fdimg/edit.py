import numpy as np
import math


def split_img_horizontally(img: np.ndarray, lsize: int) -> np.ndarray:
    h, w, c = img.shape

    if h % lsize != 0:
        print('Caution: The size of the image is not divisible by the number of layers.')

    delta = h // lsize
    layers = np.zeros((lsize, h, w, c), dtype=np.float32)

    for n in range(0, lsize):
        y_from = delta * n
        y_to = delta * (n + 1)

        layers[n, y_from:y_to] = img[y_from:y_to]

    return layers


def overlay(images: np.ndarray) -> np.ndarray:
    assert images.ndim == 4, "Dimension of images is not 4."

    lsize, height, width, color = images.shape

    result = np.zeros((height, width, color), dtype=images.dtype)
    for i in range(lsize):
        result += images[i]

    return result


def montage(images: np.ndarray):
    assert images.ndim == 4, "Dimension of images is not 4."

    lsize, n, m, c = images.shape

    column = math.ceil(math.sqrt(lsize))
    row = int(np.ceil(lsize / column))

    img = np.zeros((n * row, m * column, c), dtype=images.dtype)

    for i in range(lsize):
        iy = i // column
        ix = i - column * iy

        y = iy * n
        x = ix * m

        img[y:y+n, x:x+m] = images[i]

    return img


def convert_to_uint8(img: np.ndarray) -> np.ndarray:
    return np.clip(img.real, 0, 255).astype(np.uint8)

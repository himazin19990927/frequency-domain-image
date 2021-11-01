from typing import Tuple

from skimage import io
import numpy as np
import cv2
from matplotlib import pyplot as plt

from fdimg import fft, edit, focalstack, filtering

img = io.imread('./image/lenna.bmp')
height, width, color = img.shape

lsize = 8
sigma = 1


layers = edit.split_img_horizontally(img, lsize)

plt.subplot(1, 2, 1)
plt.imshow(edit.convert_to_uint8(edit.montage(layers)))

layers_fft = fft.fft2(layers)


fs_fft = focalstack.make_focalstack(
    layers_fft, sigma, filter_func=filtering.gaussian_psf_meshgrid)

fs_ifft = fft.ifft2(fs_fft)

plt.subplot(1, 2, 2)
plt.imshow(edit.convert_to_uint8(edit.montage(fs_ifft)))

plt.show()

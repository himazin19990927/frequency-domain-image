from typing import Tuple

from skimage import io
from matplotlib import pyplot as plt

import sys
sys.path.append("../")
from fdimg import fft, filtering, edit

img = io.imread('./image/lenna.bmp')
height, width, color = img.shape

img_fft = fft.fft2(img)

meshgrid = filtering.meshgrid(size=(height, width))
H = filtering.gaussian_psf_meshgrid(meshgrid, 7)

res_fft = fft.filter2(img_fft, H)
img_ifft = fft.ifft2(res_fft)

plt.imshow(edit.convert_to_uint8(img_ifft))

plt.show()
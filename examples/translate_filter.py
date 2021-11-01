from skimage import io
from matplotlib import pyplot as plt

import sys
sys.path.append("../")
from fdimg import fft, edit, filtering

img = io.imread('./image/lenna.bmp')
height, width, color = img.shape

img_fft = fft.fft2(img)

meshgrid = filtering.meshgrid(size=(height, width))
H = filtering.translate_meshgrid(meshgrid, dx=100, dy=50)

res_fft = fft.filter2(img_fft, H)
img_ifft = fft.ifft2(res_fft)

plt.imshow(edit.convert_to_uint8(img_ifft))
plt.show()

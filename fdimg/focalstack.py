import numpy as np
from fdimg import fft, filtering

from typing import Callable, Tuple


def make_focalstack(layers: np.ndarray, sigma: float, filter_func: Callable[[Tuple[np.ndarray, np.ndarray], float], np.ndarray]) -> np.ndarray:
    """
    Return focal stack images.
    Parameters
    ----------
    layers : numpy.ndarray of complex128
        Images array of frequency domain. It's shaped like [Layer, Height, Width, Color].
    sigma : float
        psf parameter.

    Returns
    -------
    out : numpy.ndarray of complex128
        Calculated focal stack. It's shaped like [Layer, Height, Width, Color].
    """

    lsize, height, width, color = layers.shape
    meshgrid = filtering.meshgrid(size=(height, width))

    h = np.ones((lsize, height, width), dtype=np.complex128)
    for i in range(1, lsize):
        h[i] = filter_func(meshgrid, sigma * i)

    focalstack = np.zeros((lsize, height, width, color), dtype=np.complex128)
    for m in range(lsize):
        for n in range(lsize):
            focalstack[m] += fft.filter2(layers[n], h[abs(m - n)])

    return focalstack

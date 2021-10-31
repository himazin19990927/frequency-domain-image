import numpy as np
from typing import Tuple


def meshgrid(size=Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return frequency matrices from size.

    Parameters
    ----------
    size : Tuple of ints.
        Size of meshgrid, e.g. ``size=(height, width)``

    Returns
    -------
    out : Tuple of numpy.ndarray
        Meshgrid for frequency. Frequency is a constant interval from -0.5 to 0.5

    Examples
    --------
    >>> u, v = meshgrid(size=(4, 4))
    >>> u
    array([[-0.5 , -0.25,  0.  ,  0.25],
           [-0.5 , -0.25,  0.  ,  0.25],
           [-0.5 , -0.25,  0.  ,  0.25],
           [-0.5 , -0.25,  0.  ,  0.25]])
    >>> v
    array([[-0.5 , -0.5 , -0.5 , -0.5 ],
           [-0.25, -0.25, -0.25, -0.25],
           [ 0.  ,  0.  ,  0.  ,  0.  ],
           [ 0.25,  0.25,  0.25,  0.25]])
    """

    height, width = size

    u, v = np.meshgrid(np.arange(0, width), np.arange(0, height))
    u = u / width - 0.5
    v = v / height - 0.5

    return (u, v)


def cauthy_psf_meshgrid(meshgrid: Tuple[np.ndarray, np.ndarray], sigma: float) -> np.ndarray:
    """
    Return PSF(Point Spread Function) matrix from meshgrid and parameter sigma.
    PSF is defined in the frequency domain as follows.
        A(s, t, σ) = exp(-2πσ * sqrt(s^2 + t ^ 2))

    Parameters
    ----------
    meshgrid: Tuple of numpy.ndarray
        Meshgrid for frequency, e.g.``meshgrid=(u, v)``

    Returns
    -------
    out: numpy.ndarray
        Calculated PSF matrix.
        Zero frequency components of the matrix are shifted to the edge, because the Fourier transformed image has the same shape.

    Examples
    --------
    >>> u, v = meshgrid(size=(4, 4))
    >>> h = psf_meshgrid(meshgrid=(u, v))
    >>> h
    array([[1.        , 0.20787958, 0.04321392, 0.20787958],
           [0.20787958, 0.10845266, 0.02982503, 0.10845266],
           [0.04321392, 0.02982503, 0.01176198, 0.02982503],
           [0.20787958, 0.10845266, 0.02982503, 0.10845266]])
    """

    u, v = meshgrid

    H = np.exp(-2 * np.pi * sigma * np.hypot(u, v))
    H = np.fft.fftshift(H)

    return H


def cauchy_psf(size: Tuple[int, int], sigma: float) -> np.ndarray:
    """
    Return PSF(Point Spread Function) matrix from size and parameter sigma.
    PSF is defined in the frequency domain as follows.
        A(s, t, σ) = exp(-2πσ * sqrt(s^2 + t ^ 2))

    Parameters
    ----------
    size: Tuple of ints.
        Size of result matrix, e.g.``size=(height, width)``

    Returns
    -------
    out: numpy.ndarray
        Calculated PSF matrix.
        Zero frequency components of the matrix are shifted to the edge, because the Fourier transformed image has the same shape.

    Examples
    --------
    >>> m = cauthy_psf(size=(4, 4), sigma=1)
    >>> m
    array([[1.        , 0.20787958, 0.04321392, 0.20787958],
           [0.20787958, 0.10845266, 0.02982503, 0.10845266],
           [0.04321392, 0.02982503, 0.01176198, 0.02982503],
           [0.20787958, 0.10845266, 0.02982503, 0.10845266]])
    """

    m = meshgrid(size)

    return cauthy_psf_meshgrid(m, sigma)


def gaussian_psf_meshgrid(meshgrid: Tuple[np.ndarray, np.ndarray], sigma: float) -> np.ndarray:
    u, v = meshgrid

    H = np.exp(-2 * np.pi * sigma * (u ** 2 + v ** 2))
    H = np.fft.fftshift(H)

    return H


def gaussian_psf(size: Tuple[int, int], sigma: float) -> np.ndarray:
    m = meshgrid(size)

    return gaussian_psf_meshgrid(m, sigma)


def translate_meshgrid(meshgrid: Tuple[np.ndarray, np.ndarray], dx: int = 0, dy: int = 0):
    """
    Return translate filter matrix from meshgrid and parameter dx, dy.
    Translate filter is defined in the frequency domain as follows.
        H(s, t, dx, dy) = exp{-2πj * (s * dx + u * dy)}

    Parameters
    ----------
    meshgrid: Tuple of numpy.ndarray
        Meshgrid for frequency, e.g.``meshgrid=(u, v)``
    dx: int
        Horizontal movement amount.
    dy: int
        Vertical movement amount.

    Returns
    -------
    out: numpy.ndarray
        Calculated translate matrix.
        Zero frequency components of the matrix are shifted to the edge, because the Fourier transformed image has the same shape.
    """

    u, v = meshgrid
    H = np.exp(1j * 2 * np.pi * (u * dx + v * dy))
    H = np.fft.fftshift(H)

    return H


def translate(size: Tuple[int, int], dx: int = 0, dy: int = 0):
    """
    Return translate filter matrix from size and parameter dx, dy.
    Translate filter is defined in the frequency domain as follows.
        H(s, t, dx, dy) = exp{-2πj * (s * dx + u * dy)}

    Parameters
    ----------
    size : Tuple of ints.
        Size of meshgrid, e.g. ``size=(height, width)``
    dx: int
        Horizontal movement amount.
    dy: int
        Vertical movement amount.

    Returns
    -------
    out: numpy.ndarray
        Calculated translate matrix.
        Zero frequency components of the matrix are shifted to the edge, because the Fourier transformed image has the same shape.
    """

    m = meshgrid(size)

    return translate_meshgrid(m, dx, dy)

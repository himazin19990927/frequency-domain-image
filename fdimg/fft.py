import numpy as np


def fft2(img: np.ndarray) -> np.ndarray:
    """
    Compute 2-dimentional discrete fourier transform of a color image or stacked color images.

    Parameters
    ----------
    img: numpy.ndarray
        A color image shaped like [height, width, color] or stacked color images shaped like [layer, height, width, color]
    
    Returns
    -------
    out: numpy.ndarray of numpy.complex128
        Fourier transformed image or images.
    """

    assert img.ndim == 3 or img.ndim == 4, "The dimention of argument img is not 3 or 4"

    if img.ndim == 3:
        # When img is simply image array like [height, width, color]
        height, width, color = img.shape
        result = np.zeros((height, width, color), dtype=np.complex128)

        for c in range(color):
            result[:, :, c] = np.fft.fft2(img[:, :, c])

        return result
    else:
        # When img is stacked image array like [layer, height, width, color]
        lsize, height, width, color = img.shape
        result = np.zeros((lsize, height, width, color), dtype=np.complex128)

        for l in range(lsize):
            for c in range(color):
                result[l, :, :, c] = np.fft.fft2(img[l, :, :, c])

        return result


def ifft2(img: np.ndarray) -> np.ndarray:
    """
    Compute 2-dimentional inverse discrete fourier transform of a color image or stacked color images.

    Parameters
    ----------
    img: numpy.ndarray of numpy.complex128
        A color image shaped like [height, width, color] or stacked color images shaped like [layer, height, width, color]
    
    Returns
    -------
    out: numpy.ndarray of numpy.complex128
        Inverse fourier transformed image or images.
    """

    assert img.ndim == 3 or img.ndim == 4, "The dimention of argument img is not 3 or 4"

    if img.ndim == 3:
        # When img is simply image array like [height, width, color]
        height, width, color = img.shape
        result = np.zeros((height, width, color), dtype=np.complex128)

        for c in range(color):
            result[:, :, c] = np.fft.ifft2(img[:, :, c])

        return result

    else:
        # When img is stacked image array like [Layer, Height, Width, Color]
        lsize, height, width, color = img.shape
        result = np.zeros((lsize, height, width, color), dtype=np.complex128)

        for l in range(lsize):
            for c in range(color):
                result[l, :, :, c] = np.fft.ifft2(img[l, :, :, c])

        return result

def filter2(img: np.ndarray, filter: np.ndarray) -> np.ndarray:
    """
    Apply a frequency filter to an image or stacked images.

    Parameters
    ----------
    img: numpy.ndarray
        Input image.
    filter: numpy.ndarray
        One filter or multiple filters to apply. \n
        If more than one filter is given, the same number of images must be given. \n
        Zero frequency components of the filter needs to be shifted to the edge.
    
    Returns
    -------
    out: numpy.ndarray
        Resulting image.
    """

    assert img.ndim == 3 or img.ndim == 4, "The dimention of argument 'img' must be 3 or 4"
    assert filter.ndim == 2 or filter.ndim == 3, "The dimention of argument 'filter' must be 2 or 3"

    if img.ndim == 3:
        # When img is simply image array like [Height, Width, Color]
        assert filter.ndim == 2, "While only one image was given, multiple filters were given."

        height, width, color = img.shape
        result = np.zeros((height, width, color), dtype=np.complex128)

        for c in range(color):
            result[:, :, c] = img[:, :, c] * filter

        return result
    else:
        # When img is stacked image array like [Layer, Height, Width, Color]
        lsize, height, width, color = img.shape
        result = np.zeros((lsize, height, width, color), dtype=np.complex128)

        if filter.ndim == 2:
            # If the given filter is one
            for l in range(lsize):
                for c in range(color):
                    result[l, :, :, c] = img[l, :, :, c] * filter
        else:
            # If there is more than one filter given.
            assert filter.shape[0] == lsize, "The number of filters given does not match the number of images given."

            for l in range(lsize):
                for c in range(color):
                    result[l, :, :, c] = img[l, :, :, c] * filter[l]
        
        return result

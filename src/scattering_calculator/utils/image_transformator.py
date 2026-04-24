from __future__ import annotations

import scipy as scp
from scipy.ndimage import fourier_shift
import numpy as np
from numpy.typing import ArrayLike, NDArray


def Fraunhofer_propagation_jones(E):
    """Propagate a Jones wavefield to the far field using Fourier transform.

    Parameters
    ----------
    E : ndarray of shape (Ny, Nx, 2)
        Input Jones wavefield.

    Returns
    -------
    E_far : ndarray of shape (Ny, Nx, 2)
        Far-field Jones wavefield.
    """
    return scp.fft.fftshift(scp.fft.fft2(scp.fft.ifftshift(E, axes=(0, 1)), workers=-1, axes=(0, 1)), axes=(0, 1))

def reconstruct(I):
    """FTH reconstruction of scalar hologram.

    Parameters
    ----------
    I : ndarray of shape (Ny, Nx)
        Input intensity image.

    Returns
    -------
    FTH reconstruction of the input hologram.

    """

    return scp.fft.fftshift(scp.fft.fft2(scp.fft.ifftshift(I), workers=-1))



def shift_image(
    image: NDArray[np.floating],
    shift: ArrayLike,
) -> NDArray[np.float64]:
    """Shift an image with sub-pixel precision using Fourier-space translation.

    Parameters
    ----------
    image : ndarray
        Input 2-D image to be shifted.
    shift : array-like of float
        Translation vector ``[dy, dx]`` in pixels (sub-pixel values allowed).

    Returns
    -------
    image_shifted : ndarray
        Real-valued shifted image.
    """

    # Shift Image
    shift_image = fourier_shift(scp.fft.fft2(image, workers=-1), shift)
    shift_image = scp.fft.ifft2(shift_image, workers=-1)
    shift_image = shift_image.real

    return shift_image


def binning(
    array: NDArray[np.floating],
    binning_factor: int,
) -> NDArray[np.float64]:
    """Bin an array by averaging blocks of pixels.

    The last two dimensions are reduced: ``new_shape = old_shape // binning_factor``.

    Parameters
    ----------
    array : ndarray, ndim >= 2
        Input array. Only the last two dimensions are binned.
    binning_factor : int
        Downsampling factor. Pass ``1`` to return the array unchanged.

    Returns
    -------
    new_array : ndarray
        Binned array with last two dimensions divided by ``binning_factor``.

    Raises
    ------
    ValueError
        If ``array`` has fewer than 2 dimensions.
    """

    # Only if binning factor is relevant
    if binning_factor != 1:
        if array.ndim >= 2:
            # New output shape
            new_shape = np.array(array.shape)
            new_shape[-2:] = (new_shape[-2:] // binning_factor).astype(int)

            # Reshape by adding additional dimensions
            shape = new_shape.copy()
            shape = np.insert(shape, -1, array.shape[-1] // new_shape[-1])
            shape = np.append(shape, array.shape[-2] // new_shape[-2]).astype(int)

            new_array = array.reshape(shape).mean(-1).mean(-2)
            return new_array
        else:
            raise ValueError(
                f"Dimension mismatch: input array must have at least 2 dimension"
            )

    elif binning_factor == 1:
        return array


def make_square_shape(images: NDArray[np.floating]) -> NDArray[np.floating]:
    """Crop the last two dimensions of an N-D array to a square.

    The smaller of the last two dimension sizes is used as the side length.

    Parameters
    ----------
    images : ndarray
        Input array with at least 2 dimensions.

    Returns
    -------
    images : ndarray
        Array with last two dimensions cropped to ``(min_dim, min_dim)``.
    """
    crop = np.min(np.array([images.shape[-2], images.shape[-1]]))
    images = images[..., :crop, :crop]

    return images


def hls_to_rgb(hls_array: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert an HLS colour array to RGB.

    Follows the same HLS convention as Python's ``colorsys`` module (hue,
    *lightness*, saturation — not HSL).

    Parameters
    ----------
    hls_array : ndarray of shape (H, W, 3)
        Array of HLS values, all in ``[0, 1]``.

    Returns
    -------
    rgb : ndarray of shape (H, W, 3)
        Corresponding RGB values in ``[0, 1]``.

    Notes
    -----
    Adapted from https://gist.github.com/reinhrst/2d693a16c04861a8fbc5253938312410
    """

    y, x, z = hls_array.shape
    hls_array = np.reshape(hls_array, (y * x, z))

    ONE_THIRD = 1 / 3
    TWO_THIRD = 2 / 3
    ONE_SIXTH = 1 / 6

    def _v(m1, m2, h):
        h = h % 1.0
        return np.where(
            h < ONE_SIXTH,
            m1 + (m2 - m1) * h * 6,
            np.where(
                h < 0.5,
                m2,
                np.where(h < TWO_THIRD, m1 + (m2 - m1) * (TWO_THIRD - h) * 6, m1),
            ),
        )

    assert hls_array.ndim == 2
    assert hls_array.shape[1] == 3
    assert np.max(hls_array) <= 1
    assert np.min(hls_array) >= 0

    h, l, s = hls_array.T.reshape((3, -1, 1))
    m2 = np.where(l < 0.5, l * (1 + s), l + s - (l * s))
    m1 = 2 * l - m2

    r = np.where(s == 0, l, _v(m1, m2, h + ONE_THIRD))
    g = np.where(s == 0, l, _v(m1, m2, h))
    b = np.where(s == 0, l, _v(m1, m2, h - ONE_THIRD))

    rgb = np.reshape(np.concatenate((r, g, b), axis=1), (y, x, z))
    return rgb


def complex_to_color(
    array: NDArray[np.complexfloating],
    abs_range: list[float] = [0, 100],
) -> NDArray[np.float64]:
    """Encode a complex-valued array as an RGB image using the HLS colour system.

    Phase maps to hue (colour) and magnitude maps to lightness (brightness),
    giving an intuitive visual representation of complex wavefields.

    Parameters
    ----------
    array : complex ndarray of shape (H, W)
        Input complex array.
    abs_range : list of two floats, optional
        Percentile range ``[p_low, p_high]`` used to normalise the magnitude
        to ``[0, 1]`` before mapping to lightness. Default is ``[0, 100]``.

    Returns
    -------
    rgb : ndarray of shape (H, W, 3)
        RGB image with values in ``[0, 1]``.
    """
    # In HLS color system: hue (color), lightness, saturation
    # Angle should represent color
    hue = (np.angle(array) + np.pi) / (2 * np.pi)

    # Lightness
    lightness = np.abs(array)
    vmin, vmax = np.percentile(lightness, abs_range)
    lightness = (lightness - vmin) / (vmax - vmin)

    # Clip to color range
    lightness = np.clip(lightness, a_min=0, a_max=1)

    # Saturation
    saturation = 1 * np.ones(array.shape)

    return hls_to_rgb(np.stack((hue, lightness, saturation), axis=2))

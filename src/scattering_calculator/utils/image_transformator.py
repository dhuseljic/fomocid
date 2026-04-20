import scipy as scp
from scipy.ndimage import fourier_shift
import numpy as np


def shift_image(image, shift):
    """
    Shifts image with sub-pixel precission in Fourier space


    Parameters
    ----------
    image: array
        Moving image, will be shifted by shift vector

    shift: vector
        x and y translation in px

    Returns
    -------
    image_shifted: array
        Shifted image
    -------
    author: CK 2021
    """

    # Shift Image
    shift_image = fourier_shift(scp.fft.fft2(image, workers=-1), shift)
    shift_image = scp.fft.ifft2(shift_image, workers=-1)
    shift_image = shift_image.real

    return shift_image


def binning(array, binning_factor):
    """
    Bins images: new_shape = old_shape/binning_factor

    Parameter
    =========
    array : numpy array ndim > 1
        last two dimension will be binned
    binning_factor : int
        new_shape = old_shape/binning_factor for last two dimensions

    Output
    ======
    new_array : numpy array
        binned array
    ======
    author: ck 2023/24

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


def make_square_shape(images):
    """Crops last two dimenions of n-d images to square shape"""
    crop = np.min(np.array([images.shape[-2], images.shape[-1]]))
    images = images[..., :crop, :crop]

    return images


def hls_to_rgb(hls_array: np.ndarray) -> np.ndarray:
    """
    Expects an array of shape (X, 3), each row being HLS colours.
    Returns an array of same size, each row being RGB colours.
    Like `colorsys` python module, all values are between 0 and 1.

    NOTE: like `colorsys`, this uses HLS rather than the more usual HSL

    from: https://gist.github.com/reinhrst/2d693a16c04861a8fbc5253938312410
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


def complex_to_color(array, abs_range=[0, 100]):
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

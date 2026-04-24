from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage.filters import gaussian_filter
from scipy.special import erf
from numpy.typing import NDArray

def circle_mask(
    shape: tuple[int, int],
    center: tuple[float, float],
    radius: float,
    sigma: float | None = None,
) -> NDArray[np.float64]:
    """Draw a circular mask with optional Gaussian edge smoothing.

    Parameters
    ----------
    shape : tuple of int
        Shape (rows, cols) of the output array.
    center : tuple of float
        Centre coordinates ``(y_center, x_center)`` in pixels.
    radius : float
        Radius of the mask in pixels. Diameter is always ``2*radius + 1`` px.
    sigma : float or None, optional
        Standard deviation of the Gaussian smoothing filter.
        No smoothing when ``None`` or ``0``.

    Returns
    -------
    mask : ndarray of shape ``shape``
        Binary mask, or smoothed binary mask if ``sigma`` is given.
    """

    # setup array
    x = np.linspace(0, shape[1] - 1, shape[1])
    y = np.linspace(0, shape[0] - 1, shape[0])
    X, Y = np.meshgrid(x, y)

    # define circle
    mask = np.sqrt(((X - center[1]) ** 2 + (Y - center[0]) ** 2)) <= (radius)
    mask = mask.astype(float)

    # smooth aperture
    if sigma is not None and sigma != 0:
        mask = gaussian_filter(mask, sigma)

    return mask


def create_set_of_circle_masks(
    circle_coordinates: list[list[float]],
    shape: tuple[int, int],
) -> NDArray[np.float64]:
    """Create a support mask from a combination of multiple circular apertures.

    Parameters
    ----------
    circle_coordinates : list of [y_center, x_center, radius]
        Centre coordinates and radius of each aperture in pixels.
    shape : tuple of int
        Shape (rows, cols) of the output array.

    Returns
    -------
    supportmask : ndarray of shape ``shape``
        Composite binary mask where circular apertures are ``1``.
    """

    # Create support mask
    supportmask = np.zeros(shape)
    for yc, xc, r in circle_coordinates:
        supportmask += circle_mask(supportmask.shape, [yc, xc], r)

    return supportmask


def smooth_box_1d(r: NDArray[np.float64], radius: float, sigma: float) -> NDArray[np.float64]:
    """
    Smooth radial box:
    ~1 inside radius, ~0 outside radius, with edge width controlled by sigma.
    """
    return 0.5 * (1.0 - erf((r - radius) / (np.sqrt(2.0) * sigma)))


def circle_mask3D(
    shape: tuple[int, int, int],
    center: tuple[float, float],
    radius: float,
    sigma: float | None = None,
) -> NDArray[np.float64]:
    """
    Create a circular mask on the y-x plane from a radial smooth box function.

    Parameters
    ----------
    shape : tuple[int, int, int]
        Shape of the 3D volume as (z, y, x).
    center : tuple[float, float]
        Circle center as (y_center, x_center), in pixels.
    radius : float
        Circle radius in pixels.
    sigma : float or None
        Edge smoothing width in pixels. If None or 0, returns a binary mask.

    Returns
    -------
    mask : ndarray
        2D mask with shape (y, x).
    """

    _, ny, nx = shape

    y = np.arange(ny, dtype=float)
    x = np.arange(nx, dtype=float)
    X, Y = np.meshgrid(x, y, indexing="xy")

    r = np.sqrt((X - center[1])**2 + (Y - center[0])**2)

    if sigma is None or sigma == 0:
        mask = (r <= radius).astype(float)
    else:
        mask = smooth_box_1d(r, radius, sigma)

    return mask


def create_set_of_circle_masks3D(
    circle_coordinates: list[list[float]],
    shape: tuple[int, int],
) -> NDArray[np.float64]:
    """Create a support mask from a combination of multiple circular apertures.

    Parameters
    ----------
    circle_coordinates : list of [y_center, x_center, radius]
        Centre coordinates and radius of each aperture in pixels.
    shape : tuple of int
        Shape (rows, cols) of the output array.

    Returns
    -------
    supportmask : ndarray of shape ``shape``
        Composite binary mask where circular apertures are ``1``.
    """

    # Create support mask
    supportmask = np.zeros(shape)
    for yc, xc, r in circle_coordinates:
        supportmask += circle_mask3D(supportmask.shape, [yc, xc], r)

    return supportmask

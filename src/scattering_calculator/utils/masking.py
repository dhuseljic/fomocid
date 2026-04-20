import numpy as np
from scipy.ndimage.filters import gaussian_filter


def circle_mask(shape, center, radius, sigma=None):
    """
    Draws circle mask with option to apply gaussian filter for smoothing

    Parameter
    =========
    shape : int tuple
        shape/dimension of output array
    center : int tuple
        center coordinates (ycenter,xcenter)
    radius : scalar
        radius of mask in px. Care: diameter is always (2*radius+1) px
    sigma : scalar
        std of gaussian filter

    Output
    ======
    mask: array
        binary mask, or smoothed binary mask
    ======
    author: ck 2022
    """

    # setup array
    x = np.linspace(0, shape[1] - 1, shape[1])
    y = np.linspace(0, shape[0] - 1, shape[0])
    X, Y = np.meshgrid(x, y)

    # define circle
    mask = np.sqrt(((X - center[1]) ** 2 + (Y - center[0]) ** 2)) <= (radius)
    mask = mask.astype(float)

    # smooth aperture
    if np.logical_and(sigma != None, sigma != 0):
        mask = gaussian_filter(mask, sigma)

    return mask


def create_set_of_circle_masks(circle_coordinates, shape):
    """
    Create cdi support mask from a combination of multiple circular apertures

    Parameter
    =========
    circle_coordinates: nested list
        Contains center coordinates and radius of each aperture [[yc_1,xc_1,r_1],[yc_2,xc_2,r_2],...]
    shape : int tuple
        shape/dimension of output array

    Output
    ======
    supportmask: array
        composed binary mask where circular apertures are "1"
    ======
    author: ck 2023

    """

    # Create support mask
    supportmask = np.zeros(shape)
    for i in range(len(circle_coordinates)):
        supportmask += circle_mask(
            supportmask.shape,
            [circle_coordinates[i][0], circle_coordinates[i][1]],
            circle_coordinates[i][2],
        )

    return supportmask

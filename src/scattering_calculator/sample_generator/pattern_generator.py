from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from scipy.ndimage import gaussian_filter
from scattering_calculator.utils.masking import circle_mask


def create_skyrmion_pattern(
    sz_array: list[int],
    skyr_diameter: float,
    screening_diameter: float,
    number_skyr: int,
    number_iter: int,
    sigma: float | str = "none",
    plot: bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.int_]]:
    """Create a random skyrmion pattern using Poisson-disk-like placement.

    Parameters
    ----------
    sz_array : list of int
        Output array shape ``[rows, cols]`` in pixels.
    skyr_diameter : float
        Diameter of individual skyrmions in pixels.
    screening_diameter : float
        Minimum centre-to-centre distance between any two skyrmions in pixels.
    number_skyr : int
        Target number of skyrmions to place.
    number_iter : int
        Maximum number of placement attempts.
    sigma : float or "none", optional
        Standard deviation of the Gaussian smoothing filter applied after
        placement. Pass ``"none"`` to skip smoothing. Default is ``"none"``.
    plot : bool, optional
        If ``True``, plot the screening kernel and the final pattern.
        Default is ``False``.

    Returns
    -------
    pattern : ndarray of shape ``sz_array``
        Normalised skyrmion pattern with values in ``[-1, 1]``.
    coordinates : ndarray of shape ``(N, 2)``
        Centre coordinates of placed skyrmions, sorted by row index.
    """

    # Screening
    sz = np.ceil(screening_diameter).astype(int)
    Screening = circle_mask(
        (sz + 1, sz + 1), (0.5 * sz, 0.5 * sz), np.floor(0.5 * screening_diameter)
    )

    sky_px = Screening.shape[0]

    # Calc Skyrmion pattern
    Skyrmion = circle_mask(
        (sz + 1, sz + 1), (0.5 * sz, 0.5 * sz), np.floor(0.5 * skyr_diameter)
    )

    # Initialize
    PATTERN = np.zeros(
        (sz_array[0] + sky_px, sz_array[1] + sky_px)
    )  # Size is larger to also allow skyrmions on the edge
    Skyr_Counter = 0
    it = 0

    # Store also center coordinates of skyrmions
    coordinates = []

    # Generation loop
    print("Calculating pattern with %.0d skyrmions..." % number_skyr)
    pbar = tqdm(total=number_skyr)
    while (Skyr_Counter < number_skyr) & (it < number_iter):
        it = it + 1

        # Get random x,y coordinates
        txr = 1 + np.floor(np.random.uniform(0, 1, 1) * (sz_array[1]))
        txr = txr[0].astype(int)
        yr = 1 + np.floor(np.random.uniform(0, 1, 1) * (sz_array[0]))
        yr = yr[0].astype(int)

        # Check if there is no Skyrmion in screening distance (no overlap)
        if (
            np.sum(
                np.logical_and(
                    PATTERN[np.s_[txr : txr + sky_px, yr : yr + sky_px]], Screening
                )
            )
            == 0
        ):
            # Write Skyrmion into pattern
            PATTERN[txr : txr + sky_px, yr : yr + sky_px] = (
                PATTERN[txr : txr + sky_px, yr : yr + sky_px] + Skyrmion
            )
            Skyr_Counter = Skyr_Counter + 1
            coordinates.append([txr, yr])
            pbar.update(1)

    pbar.close()
    print("Created %d skyrmions in %d iterations!" % (Skyr_Counter, it))

    # Smoothing
    if sigma != None:
        # Smooth Pattern with gaussian filter
        PATTERN = gaussian_filter(PATTERN, float(sigma))

    # Normalize to mz in [-1,1]
    PATTERN = PATTERN / np.max(PATTERN)
    PATTERN = 2 * (PATTERN - 0.5)
    PATTERN = -PATTERN

    # Crop Pattern to original size
    PATTERN = PATTERN[
        np.floor(sky_px / 2).astype(int) : sz_array[0]
        + np.floor(sky_px / 2).astype(int),
        np.floor(sky_px / 2).astype(int) : sz_array[0]
        + np.floor(sky_px / 2).astype(int),
    ]

    # Reorder skyrmion coordinates
    coordinates = np.array(coordinates)
    ind = np.argsort(coordinates[:, 0])
    coordinates = coordinates[ind]

    # Plot
    if plot is True:
        fig, ax = plt.subplots(1, 2, figsize=(8, 4))
        ax[0].imshow(Screening + Skyrmion, cmap="gray")
        ax[0].set_title("Screening sanity check")
        ax[1].imshow(PATTERN, vmin=-1, vmax=1, cmap="gray")
        ax[1].set_title("Skyrmion Pattern")
        plt.tight_layout()

    return PATTERN, coordinates

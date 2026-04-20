import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from scipy.ndimage import gaussian_filter
from scattering_calculator.utils.masking import circle_mask


def create_skyrmion_pattern(
    sz_array,
    skyr_diameter,
    screening_diameter,
    number_skyr,
    number_iter,
    sigma="none",
    plot=False,
):
    """
    Creates random skyrmion pattern

    Parameter
    =========
    sz_array : list
        shape of output array [vert,horz]
    skyr_diameter : scalar
        diameter of skyrmions in px
    screening_diameter : scalar
        minimum distance between two skyrmions in px
    number_skyr : int
        number of created skyrmions
    number_iter : int
        number of maximal interations
    smoothing : bool
        Activate or deactive smoothing of skyrmions
    sigma: scalar
        standard deviation (?) of gaussian filter for smoothing of skyrmions, if sigma == 'none' no filter will be applied
    plot: bool
        if true magnetic pattern will be plotted

    Output
    ======
    pattern : array
        Random skyrmion pattern
    ======
    author: ck 2022/23
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

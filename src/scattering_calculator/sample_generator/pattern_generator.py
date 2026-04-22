from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from scipy.ndimage import gaussian_filter
from scipy.spatial import KDTree
from scattering_calculator.utils.masking import circle_mask


def create_skyrmion_pattern(
    sz_array: list[int],
    skyr_radius: float,
    screening_radius: float,
    number_skyr: int,
    number_iter: int,
    sigma: float | None = None,
    plot: bool = False,
    seed: int | None = None,
    batch_size: int = 256,
) -> tuple[NDArray[np.float64], NDArray[np.int_]]:
    """Create a random skyrmion pattern using batched placement and KDTree screening.

    Functionally identical to :func:`create_skyrmion_pattern` but significantly
    faster for dense patterns. Candidates are generated in batches and overlap
    is checked with a KDTree distance query (O(log n)) instead of a 2-D mask
    scan per candidate.

    Parameters
    ----------
    sz_array : list of int
        Output array shape ``[rows, cols]`` in pixels.
    skyr_radius : float
        Radius of individual skyrmions in pixels.
    screening_radius : float
        Minimum centre-to-centre distance between any two skyrmions in pixels.
    number_skyr : int
        Target number of skyrmions to place.
    number_iter : int
        Maximum total number of candidates generated before giving up.
    sigma : float or None, optional
        Standard deviation of the Gaussian smoothing filter. No smoothing when
        ``None``. Default is ``None``.
    plot : bool, optional
        If ``True``, plot the screening kernel and the final pattern.
        Default is ``False``.
    seed : int or None, optional
        Random seed for reproducibility. Default is ``None``.
    batch_size : int, optional
        Number of candidate positions generated per batch. Default is ``256``.

    Returns
    -------
    pattern : ndarray of shape ``sz_array``
        Normalised skyrmion pattern with values in ``[-1, 1]``.
    coordinates : ndarray of shape ``(N, 2)``
        Centre coordinates ``(row, col)`` of placed skyrmions, sorted by row.
    """
    rng = np.random.default_rng(seed)

    sz = np.ceil(2 * screening_radius).astype(int)
    sky_px = sz + 1

    screening_mask = circle_mask(
        (sky_px, sky_px), (0.5 * sz, 0.5 * sz), np.floor(screening_radius)
    )
    skyrmion_kernel = circle_mask(
        (sky_px, sky_px), (0.5 * sz, 0.5 * sz), np.floor(skyr_radius)
    )

    pattern = np.zeros((sz_array[0] + sky_px, sz_array[1] + sky_px))
    placed: list[list[int]] = []
    tree: KDTree | None = None
    n_iter = 0

    print(f"Placing {number_skyr} skyrmions (fast)...")
    pbar = tqdm(total=number_skyr)

    while len(placed) < number_skyr and n_iter < number_iter:
        n_candidates = min(batch_size, number_iter - n_iter)
        candidates = np.column_stack(
            [
                rng.integers(1, sz_array[0] + 1, size=n_candidates),
                rng.integers(1, sz_array[1] + 1, size=n_candidates),
            ]
        )
        n_iter += n_candidates

        # Pre-filter entire batch against already-placed skyrmions in one query
        if tree is not None:
            distances, _ = tree.query(candidates, k=1)
            candidates = candidates[distances >= 2 * screening_radius]

        # Greedily accept candidates, checking within-batch conflicts
        batch_placed: list[list[int]] = []
        for row, col in candidates:
            if len(placed) >= number_skyr:
                break
            too_close = any(
                np.sqrt((row - pr) ** 2 + (col - pc) ** 2) < 2 * screening_radius
                for pr, pc in batch_placed
            )
            if too_close:
                continue
            pattern[row : row + sky_px, col : col + sky_px] += skyrmion_kernel
            placed.append([row, col])
            batch_placed.append([row, col])
            pbar.update(1)

        # Rebuild tree once per batch, not per placement
        if placed:
            tree = KDTree(placed)

    pbar.close()
    print(f"Placed {len(placed)} skyrmions in {n_iter} iterations.")

    if sigma is not None:
        pattern = gaussian_filter(pattern, sigma)

    max_val = np.max(pattern)
    if max_val > 0:
        pattern = pattern / max_val
    pattern = -(2 * (pattern - 0.5))

    half = np.floor(sky_px / 2).astype(int)
    pattern = pattern[half : half + sz_array[0], half : half + sz_array[1]]

    coordinates = np.array(placed) if placed else np.empty((0, 2), dtype=int)
    if coordinates.shape[0] > 0:
        coordinates = coordinates[np.argsort(coordinates[:, 0])]

    if plot:
        _, ax = plt.subplots(1, 2, figsize=(8, 4))
        ax[0].imshow(screening_mask + skyrmion_kernel, cmap="gray")
        ax[0].set_title("Screening sanity check")
        ax[1].imshow(pattern, vmin=-1, vmax=1, cmap="gray")
        ax[1].set_title("Skyrmion pattern")

    return pattern, coordinates


def create_lattice(sample, scattering, setup, plot=True):
    """
    creates lattice coordinates in real and pixel space based on the lattice constant, filling and shift parameters defined in the sample xarray. The lattice points are stored in a dict called lattice. The function also plots the lattice points and the distribution of random shifts if plot is set to True.

    Parameter
    =========
    scattering : xarray
        Stores information about illumination, substrate, front wave, back wave and an optional reference mask
    sample : xarray
        Stores information about lattice like lattice constant, random shift of lattice points etc.
    plot: bool
        if true lattice will be plotted as array

    Output
    ======
    lattice : dict
        Stores lattice coordinates in px and real space coordinates
    ======
    author: ck 2023
    """

    # Setup lattice dict
    lattice = {}

    # Takeover lattice constant
    lattice["lattice_constant"] = float(sample["lattice_constant"].values.copy())

    # Number of lattice sites
    lattice["nr_sites"] = int(
        (scattering["x"][-1] - scattering["x"][0]) / lattice["lattice_constant"]
    )

    # this needs to be an even number to start with respect to center pixel
    if lattice["nr_sites"] % 2 == 1:
        lattice["nr_sites"] = lattice["nr_sites"] - 1

    # Create real space coordinates of lattice in x and y direction
    lattice["x"] = (
        np.arange(-lattice["nr_sites"] / 2, lattice["nr_sites"] / 2 + 1, 1)
        * lattice["lattice_constant"]
    )
    lattice["y"] = lattice["x"].copy()

    # Combine all different combinations in coordinates
    lattice["coordinates"] = np.zeros((len(lattice["y"]) * len(lattice["x"]), 2))
    it = 0
    for i in range(len(lattice["y"])):
        # y coordinate
        y = lattice["y"][i]

        for k in range(len(lattice["x"])):
            # x coordinate
            x = lattice["x"][k]

            # Combine in coordinates variable
            lattice["coordinates"][it, 0] = y
            lattice["coordinates"][it, 1] = x

            it = it + 1

    # Remove random lattice points
    if sample["lattice_filling"] != 1:
        idx = np.random.choice(
            np.arange(0, len(lattice["coordinates"]), 1),
            size=np.floor(
                len(lattice["coordinates"]) * sample["lattice_filling"].values
            ).astype(int),
        )
        lattice["coordinates"] = lattice["coordinates"][idx]

    # Add random shift in x and y direction
    shift_y = 0
    shift_x = 0
    if sample["lattice_shift"] != "none":
        # Create gaussian random shift
        shift_y = np.random.normal(
            loc=0,
            scale=sample["lattice_shift"].values / (2 * np.sqrt(2 * np.log(2))),
            size=len(lattice["coordinates"][:, 0]),
        )
        shift_x = np.random.normal(
            loc=0,
            scale=sample["lattice_shift"].values / (2 * np.sqrt(2 * np.log(2))),
            size=len(lattice["coordinates"][:, 1]),
        )

        # Add shift
        lattice["coordinates"][:, 0] = lattice["coordinates"][:, 0] + shift_y
        lattice["coordinates"][:, 1] = lattice["coordinates"][:, 1] + shift_x

    # Find closest matching pixel real space coordinates from real space coordinate pixel grid
    lattice["coordinates_px"] = np.zeros(lattice["coordinates"].shape, dtype=int)
    for i in range(lattice["coordinates_px"].shape[0]):
        lattice["coordinates_px"][i, 0] = np.argmin(
            np.abs(lattice["coordinates"][i, 0] - scattering["y"].values)
        )
        lattice["coordinates_px"][i, 1] = np.argmin(
            np.abs(lattice["coordinates"][i, 1] - scattering["x"].values)
        )

    # Plot full array
    tmp = np.zeros((setup["sz"], setup["sz"]))
    tmp[lattice["coordinates_px"][:, 0], lattice["coordinates_px"][:, 1]] = 1
    lattice["point_array"] = tmp.copy()

    print("Created %d lattice points!" % len(lattice["coordinates_px"]))

    # Plot
    if plot is True:
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        ax[0].set_title("Lattice points")
        ax[0].imshow(lattice["point_array"], extent=setup["extent_real"])
        ax[0].set_xlabel("x in µm")
        ax[0].set_ylabel("y in µm")
        ax[1].hist(np.ceil(shift_x / setup["px"]) * setup["px"], 50)
        ax[1].set_xlabel("Shift in µm")
        ax[1].set_title("Shift x")
        ax[2].hist(np.ceil(shift_y / setup["px"]) * setup["px"], 50)
        ax[2].set_xlabel("Shift in µm")
        ax[2].set_title("Shift y")

        plt.tight_layout()

    return lattice

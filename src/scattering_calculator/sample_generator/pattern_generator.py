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
    real_space_pixel_size: float = 1,
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

    if real_space_pixel_size != 1:
        sample_y = (np.arange(sz_array[0]) - sz_array[0] / 2) * real_space_pixel_size
        sample_x = (np.arange(sz_array[1]) - sz_array[1] / 2) * real_space_pixel_size
        scale = 1e6
    else:
        sample_y = np.arange(sz_array[0])
        sample_x = np.arange(sz_array[1])
        scale = 1

    extent_real = scale * np.array(
        [
            sample_x[0],
            sample_x[-1],
            sample_y[0],
            sample_y[-1],
        ]
    )

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
        ax[1].imshow(pattern, vmin=-1, vmax=1, cmap="gray", extent=extent_real)
        ax[1].set_title("Skyrmion pattern")

        if real_space_pixel_size != 1:
            ax[1].set_xlabel("x in µm")
            ax[1].set_ylabel("y in µm")

    return pattern, coordinates


def create_lattice(
    shape: tuple[int, int],
    real_space_pixel_size: float,
    lattice_constant: float | tuple[float, float],
    lattice_filling: float,
    lattice_shift: float | None,
    plot: bool = True,
    seed: int | None = None,
) -> dict:
    """Create a 2-D rectangular lattice of sites in real and pixel space.

    Parameters
    ----------
    shape : tuple of int
        Array shape ``(rows, cols)`` in pixels.
    real_space_pixel_size : float
        Physical size of one pixel in metres.
    lattice_constant : float or tuple of float
        Centre-to-centre distance between neighbouring sites in metres.
        Pass a single ``float`` for a square lattice, or a tuple
        ``(constant_y, constant_x)`` for a rectangular lattice with
        different spacings along each axis.
    lattice_filling : float
        Fraction of lattice sites to keep, in ``(0, 1]``. Pass ``1`` to keep
        all sites.
    lattice_shift : float or None
        FWHM of the Gaussian positional disorder applied to each site in
        metres. Pass ``None`` for a perfect lattice.
    plot : bool, optional
        If ``True``, plot the lattice point array and shift histograms.
        Default is ``True``.
    seed : int or None, optional
        Random seed for reproducibility of filling and shift. Default is
        ``None``.

    Returns
    -------
    lattice : dict
        Dictionary with keys:

        ``"lattice_constant"``, ``"lattice_filling"``, ``"lattice_shift"``
            Input parameters stored for reference.
        ``"sample_y"``, ``"sample_x"`` : ndarray
            Real-space coordinate axes of the pixel grid in metres.
        ``"extent_real"`` : ndarray of shape (4,)
            ``[x_min, x_max, y_min, y_max]`` in metres for use with
            ``imshow(extent=...)``.
        ``"x"``, ``"y"`` : ndarray
            Real-space lattice axes in metres.
        ``"coordinates"`` : ndarray of shape ``(N, 2)``
            Real-space ``(y, x)`` positions of occupied sites in metres,
            after filling and disorder.
        ``"coordinates_px"`` : ndarray of shape ``(N, 2)``
            Nearest-pixel ``(row, col)`` indices for each occupied site.
        ``"point_array"`` : ndarray of shape ``shape``
            Binary image with ``1`` at each occupied lattice site.
    """
    rng = np.random.default_rng(seed)

    # Unpack lattice constants for each axis
    if isinstance(lattice_constant, tuple) or isinstance(lattice_constant, list):
        lc_y, lc_x = lattice_constant
    else:
        lc_y = lc_x = lattice_constant

    lattice: dict = {
        "lattice_constant": lattice_constant,
        "lattice_filling": lattice_filling,
        "lattice_shift": lattice_shift,
    }

    # Real-space coordinate axes of the pixel grid
    lattice["sample_y"] = (np.arange(shape[0]) - shape[0] / 2) * real_space_pixel_size
    lattice["sample_x"] = (np.arange(shape[1]) - shape[1] / 2) * real_space_pixel_size
    lattice["extent_real"] = np.array(
        [
            lattice["sample_x"][0],
            lattice["sample_x"][-1],
            lattice["sample_y"][0],
            lattice["sample_y"][-1],
        ]
    )

    # Number of lattice sites along each axis (force even for centre symmetry)
    nr_sites_y = shape[0] * real_space_pixel_size / lc_y
    if nr_sites_y % 2 == 1:
        nr_sites_y -= 1

    nr_sites_x = shape[1] * real_space_pixel_size / lc_x
    if nr_sites_x % 2 == 1:
        nr_sites_x -= 1

    axis_y = np.arange(-nr_sites_y / 2, nr_sites_y / 2 + 1) * lc_y
    axis_x = np.arange(-nr_sites_x / 2, nr_sites_x / 2 + 1) * lc_x
    lattice["y"] = axis_y
    lattice["x"] = axis_x

    # All (y, x) site combinations via meshgrid
    grid_y, grid_x = np.meshgrid(axis_y, axis_x, indexing="ij")
    lattice["coordinates"] = np.column_stack([grid_y.ravel(), grid_x.ravel()])

    # Random sub-filling
    if lattice_filling != 1:
        n_keep = int(np.floor(len(lattice["coordinates"]) * lattice_filling))
        idx = rng.choice(len(lattice["coordinates"]), size=n_keep, replace=False)
        lattice["coordinates"] = lattice["coordinates"][idx]

    # Gaussian positional disorder
    shift_y = np.zeros(len(lattice["coordinates"]))
    shift_x = np.zeros(len(lattice["coordinates"]))
    if lattice_shift is not None:
        sigma_shift = lattice_shift / (2 * np.sqrt(2 * np.log(2)))
        shift_y = rng.normal(0, sigma_shift, size=len(lattice["coordinates"]))
        shift_x = rng.normal(0, sigma_shift, size=len(lattice["coordinates"]))
        lattice["coordinates"][:, 0] += shift_y
        lattice["coordinates"][:, 1] += shift_x

    # Vectorised nearest-pixel lookup — replaces per-row loop
    lattice["coordinates_px"] = np.column_stack(
        [
            np.argmin(
                np.abs(lattice["coordinates"][:, 0:1] - lattice["sample_y"]), axis=1
            ),
            np.argmin(
                np.abs(lattice["coordinates"][:, 1:2] - lattice["sample_x"]), axis=1
            ),
        ]
    )

    point_array = np.zeros(shape)
    point_array[lattice["coordinates_px"][:, 0], lattice["coordinates_px"][:, 1]] = 1
    lattice["point_array"] = point_array

    print(f"Created {len(lattice['coordinates_px'])} lattice points.")

    if plot:
        _, ax = plt.subplots(1, 3, figsize=(12, 4))
        ax[0].set_title("Lattice points")
        ax[0].imshow(lattice["point_array"], extent=1e6 * lattice["extent_real"])
        ax[0].set_xlabel("x in µm")
        ax[0].set_ylabel("y in µm")
        ax[1].hist(np.ceil(shift_x / real_space_pixel_size) * real_space_pixel_size, 50)
        ax[1].set_xlabel("Shift in µm")
        ax[1].set_title("Shift x")
        ax[2].hist(np.ceil(shift_y / real_space_pixel_size) * real_space_pixel_size, 50)
        ax[2].set_xlabel("Shift in µm")
        ax[2].set_title("Shift y")

    return lattice

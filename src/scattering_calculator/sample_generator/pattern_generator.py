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
        Maximum number of placement attempts before giving up.
    sigma : float or None, optional
        Standard deviation of the Gaussian smoothing filter applied after
        placement. No smoothing when ``None``. Default is ``None``.
    plot : bool, optional
        If ``True``, plot the screening kernel and the final pattern.
        Default is ``False``.
    seed : int or None, optional
        Random seed for reproducibility. Default is ``None``.

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
        (sky_px, sky_px), (0.5 * sz, 0.5 * sz), np.floor(0.5 * skyr_radius)
    )

    # Pad pattern to allow placement at the edges
    pattern = np.zeros((sz_array[0] + sky_px, sz_array[1] + sky_px))
    n_placed = 0
    n_iter = 0
    coordinates = []

    print(f"Placing {number_skyr} skyrmions...")
    pbar = tqdm(total=number_skyr)

    while n_placed < number_skyr and n_iter < number_iter:
        n_iter += 1

        row = int(rng.integers(1, sz_array[0] + 1))
        col = int(rng.integers(1, sz_array[1] + 1))

        roi = pattern[row : row + sky_px, col : col + sky_px]
        if not np.any(np.logical_and(roi, screening_mask)):
            pattern[row : row + sky_px, col : col + sky_px] += skyrmion_kernel
            coordinates.append([row, col])
            n_placed += 1
            pbar.update(1)

    pbar.close()
    print(f"Placed {n_placed} skyrmions in {n_iter} iterations.")

    if sigma is not None:
        pattern = gaussian_filter(pattern, sigma)

    # Normalise to [-1, 1]
    max_val = np.max(pattern)
    if max_val > 0:
        pattern = pattern / max_val
    pattern = -(2 * (pattern - 0.5))

    # Crop back to requested size
    half = np.floor(sky_px / 2).astype(int)
    pattern = pattern[half : half + sz_array[0], half : half + sz_array[1]]

    coordinates = np.array(coordinates) if coordinates else np.empty((0, 2), dtype=int)
    if coordinates.shape[0] > 0:
        coordinates = coordinates[np.argsort(coordinates[:, 0])]

    if plot:
        _, ax = plt.subplots(1, 2, figsize=(8, 4))
        ax[0].imshow(screening_mask + skyrmion_kernel, cmap="gray")
        ax[0].set_title("Screening sanity check")
        ax[1].imshow(pattern, vmin=-1, vmax=1, cmap="gray")
        ax[1].set_title("Skyrmion pattern")

    return pattern, coordinates


def create_skyrmion_pattern_fast(
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
        rows = rng.integers(1, sz_array[0] + 1, size=n_candidates)
        cols = rng.integers(1, sz_array[1] + 1, size=n_candidates)
        n_iter += n_candidates

        for row, col in zip(rows, cols):
            if len(placed) >= number_skyr:
                break
            if tree is not None:
                dist, _ = tree.query([row, col], k=1)
                if dist < screening_radius:
                    continue
            pattern[row : row + sky_px, col : col + sky_px] += skyrmion_kernel
            placed.append([row, col])
            tree = KDTree(placed)
            pbar.update(1)

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

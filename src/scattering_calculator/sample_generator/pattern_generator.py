"""Magnetic-pattern generation helpers for simulated samples."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from scipy.ndimage import gaussian_filter, zoom
from scipy.spatial import KDTree
from scattering_calculator.sample_generator.gray_scott_generator_binary import (
    generate as generate_binary,
)
from scattering_calculator.utils.masking import circle_mask


def map_magnetization_to_3d(
    magnetic_pattern_x: NDArray[np.float64] | None = None,
    magnetic_pattern_y: NDArray[np.float64] | None = None,
    magnetic_pattern_z: NDArray[np.float64] | None = None,
    nr_repeats: int | None = None,
) -> tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]
]:
    """Combine three 2-D scalar magnetisation patterns into 3-D vector components.

    Parameters
    ----------
    magnetic_pattern_z : ndarray of shape (rows, cols)
        2-D array with values in [-1, 1] representing the z-component of the magnetisation pattern.
    magnetic_pattern_y : ndarray of shape (rows, cols)
        2-D array with values in [-1, 1] representing the y-component of the magnetisation pattern.
    magnetic_pattern_x : ndarray of shape (rows, cols)
        2-D array with values in [-1, 1] representing the x-component of the magnetisation pattern.

    Returns
    -------
    mz, my, mx : ndarray of shape (rows, cols)
        3-D vector components of the magnetisation pattern.
    """

    shapes = [
        a.shape
        for a in (magnetic_pattern_x, magnetic_pattern_y, magnetic_pattern_z)
        if a is not None
    ]

    if not shapes:
        raise ValueError(
            "At least one of magnetic_pattern_z, magnetic_pattern_y, or "
            "magnetic_pattern_x must be provided."
        )

    if len(set(shapes)) > 1:
        raise ValueError(
            f"All provided arrays must have the same shape, got shapes: {shapes}"
        )

    ref_shape = shapes[0]
    mz = magnetic_pattern_z if magnetic_pattern_z is not None else np.zeros(ref_shape)
    my = magnetic_pattern_y if magnetic_pattern_y is not None else np.zeros(ref_shape)
    mx = magnetic_pattern_x if magnetic_pattern_x is not None else np.zeros(ref_shape)

    magnetization = np.stack((mx, my, mz))
    magnetization = np.transpose(magnetization, (1, 2, 0))

    if nr_repeats is not None:
        magnetization = np.broadcast_to(
            magnetization[np.newaxis, ...], (nr_repeats, *magnetization.shape)
        )

    return magnetization


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


def skyrmions_on_lattice(
    shape: tuple[int, int],
    lattice: dict,
    skyr_radius: float,
    skyr_smoothing: float,
    diameter_spread_fwhm: float | None,
    real_space_pixel_size: float,
    plot: bool = True,
) -> NDArray[np.float64]:
    """Place skyrmions on a pre-computed lattice and return the magnetisation pattern.

    Parameters
    ----------
    shape : tuple of int
        Output array shape ``(rows, cols)`` in pixels.
    lattice : dict
        Lattice dictionary returned by :func:`create_lattice`. Updated
        in-place with a ``"skyr_radius"`` key (per-site radii in pixels).
    skyr_radius : float
        Nominal skyrmion radius in pixels.
    skyr_smoothing : float
        Standard deviation of the Gaussian smoothing filter in pixels.
        Pass ``0`` to skip smoothing.
    diameter_spread_fwhm : float or None
        FWHM of the Gaussian diameter disorder in metres. Each site gets an
        independent radius drawn from ``Normal(skyr_radius, sigma_px)`` where
        ``sigma_px = diameter_spread_fwhm / real_space_pixel_size / (2*sqrt(2*ln2))``.
        Pass ``None`` for uniform radii.
    real_space_pixel_size : float
        Physical size of one pixel in metres. Used only when
        ``diameter_spread_fwhm`` is not ``None``.
    plot : bool, optional
        If ``True``, plot the single skyrmion kernel, the full pattern, and
        the diameter histogram. Default is ``True``.

    Returns
    -------
    pattern : ndarray of shape ``shape``
        Normalised magnetisation pattern with values in ``[-1, 1]``.
    """
    n_sites = lattice["coordinates_px"].shape[0]

    if diameter_spread_fwhm is None:
        lattice["skyr_radius"] = np.full(n_sites, skyr_radius)
    else:
        scale_px = (
            diameter_spread_fwhm / real_space_pixel_size / (2 * np.sqrt(2 * np.log(2)))
        )
        lattice["skyr_radius"] = np.random.normal(
            loc=skyr_radius, scale=scale_px, size=n_sites
        )

    pattern_ext = np.ceil(2 * np.max(lattice["skyr_radius"])).astype(int) + 2

    pattern = np.zeros((shape[0] + 2 * pattern_ext, shape[1] + 2 * pattern_ext))

    skyrmion_kernel = None
    for i in range(n_sites):
        r = lattice["skyr_radius"][i]
        sz = np.ceil(2 * r).astype(int)
        skyrmion_kernel = circle_mask(
            (sz + 1, sz + 1), (0.5 * sz, 0.5 * sz), np.floor(r)
        )
        x = lattice["coordinates_px"][i, 0] + pattern_ext
        y = lattice["coordinates_px"][i, 1] + pattern_ext
        pattern[
            x : x + skyrmion_kernel.shape[0], y : y + skyrmion_kernel.shape[1]
        ] += skyrmion_kernel

    sigma = float(skyr_smoothing)
    if sigma != 0:
        pattern = gaussian_filter(pattern, sigma)
        max_val = np.max(pattern)
        if max_val > 0:
            pattern = pattern / max_val
        pattern = -(2 * (pattern - 0.5))

    pattern = pattern[
        pattern_ext : shape[0] + pattern_ext,
        pattern_ext : shape[1] + pattern_ext,
    ]

    if plot:
        _, ax = plt.subplots(1, 3, figsize=(12, 4))
        if skyrmion_kernel is not None:
            half_nm = skyrmion_kernel.shape[0] / 2 * real_space_pixel_size * 1e9
            ax[0].imshow(
                skyrmion_kernel,
                cmap="gray",
                extent=[-half_nm, half_nm, half_nm, -half_nm],
            )
            ax[0].set_xlabel("x in nm")
            ax[0].set_ylabel("y in nm")
        ax[0].set_title("Single Binary Skyrmion")
        ax[1].imshow(pattern, cmap="gray", extent=1e6 * lattice["extent_real"])
        ax[1].set_title("Skyrmion lattice")
        ax[1].set_xlabel("x in µm")
        ax[1].set_ylabel("y in µm")
        ax[2].hist(np.ceil(lattice["skyr_radius"]) * real_space_pixel_size * 1e9, 50)
        ax[2].set_title("Skyrmion Diameter")
        ax[2].set_xlabel("Diameter in nm")

    return pattern


def _rough_ellipse_mask(
    radius_y: float,
    radius_x: float,
    angle: float = 0.0,
    roughness: float = 0.0,
    roughness_modes: tuple[int, int] = (0, 0),
    rng: np.random.Generator | None = None,
) -> NDArray[np.float64]:
    """Return a local binary mask for one elliptical, rough skyrmion core.

    Parameters
    ----------
    radius_y : float
        Input value for ``radius_y``.
    radius_x : float
        Input value for ``radius_x``.
    angle : float
        Input value for ``angle``.
    roughness : float
        Input value for ``roughness``.
    roughness_modes : tuple[int, int]
        Input value for ``roughness_modes``.
    rng : np.random.Generator | None
        Input value for ``rng``.

    Returns
    -------
    result : NDArray[np.float64]
        Return value produced by the function.
    """
    rng = np.random.default_rng() if rng is None else rng
    radius_y = max(float(radius_y), 0.5)
    radius_x = max(float(radius_x), 0.5)
    roughness = max(0.0, float(roughness))
    roughness_scale = 1.0 + 2.0 * roughness
    half_y = int(np.ceil(radius_y * roughness_scale)) + 3
    half_x = int(np.ceil(radius_x * roughness_scale)) + 3

    y = np.arange(-half_y, half_y + 1)
    x = np.arange(-half_x, half_x + 1)
    yy, xx = np.meshgrid(y, x, indexing="ij")

    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    x_rot = xx * cos_a + yy * sin_a
    y_rot = -xx * sin_a + yy * cos_a
    normalized_radius = np.sqrt((y_rot / radius_y) ** 2 + (x_rot / radius_x) ** 2)

    boundary = 1.0
    if roughness > 0:
        polar_angle = np.arctan2(y_rot / radius_y, x_rot / radius_x)
        min_mode, max_mode = roughness_modes
        if max_mode >= min_mode and min_mode >= 1:
            boundary = np.ones_like(normalized_radius)
            for mode in range(int(min_mode), int(max_mode) + 1):
                amplitude = rng.normal(scale=roughness / mode)
                phase = rng.uniform(0, 2 * np.pi)
                boundary += amplitude * np.cos(mode * polar_angle + phase)
            boundary = np.clip(boundary, 1.0 - 2.0 * roughness, 1.0 + 2.0 * roughness)

    return (normalized_radius <= boundary).astype(float)


def create_disordered_skyrmion_lattice_pattern(
    sz_array: list[int] | tuple[int, int],
    stripe_width: float,
    sigma: float | None = None,
    skyrmion_density: float = 0.25,
    diameter_spread: float = 0.0,
    ellipticity: tuple[float, float] = (0.75, 1.35),
    roughness: float = 0.05,
    roughness_modes: tuple[int, int] = (3, 9),
    positional_disorder: float = 0.0,
    placement_center: tuple[float, float] | None = None,
    placement_radius: float | None = None,
    seed: int | None = None,
    plot: bool = False,
    real_space_pixel_size: float = 1,
    **_,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Create a random non-overlapping distribution of irregular skyrmions.

    ``stripe_width`` is interpreted as the average skyrmion diameter in pixels.
    ``diameter_spread`` and ``sigma`` are also pixel lengths by the time this
    function is called; pipeline configs can specify them in metres and
    :class:`MagneticPatternConfig` converts them. ``diameter_spread`` is a
    bounded half-range, so diameters are sampled from
    ``diameter +/- diameter_spread`` instead of an unbounded Gaussian. The
    skyrmion density is an approximate target area fraction of skyrmion cores
    in the generated field. If ``placement_radius`` is provided, random
    candidate centers are drawn inside that circular placement region, normally
    the OH radius plus one skyrmion diameter. Candidate centers are accepted
    greedily only when their conservative bounding radii do not overlap any
    previously accepted skyrmion.

    Parameters
    ----------
    sz_array : list[int] | tuple[int, int]
        Input value for ``sz_array``.
    stripe_width : float
        Input value for ``stripe_width``.
    sigma : float | None
        Input value for ``sigma``.
    skyrmion_density : float
        Input value for ``skyrmion_density``.
    diameter_spread : float
        Input value for ``diameter_spread``.
    ellipticity : tuple[float, float]
        Input value for ``ellipticity``.
    roughness : float
        Input value for ``roughness``.
    roughness_modes : tuple[int, int]
        Input value for ``roughness_modes``.
    positional_disorder : float
        Input value for ``positional_disorder``.
    placement_center : tuple[float, float] | None
        Input value for ``placement_center``.
    placement_radius : float | None
        Input value for ``placement_radius``.
    seed : int | None
        Input value for ``seed``.
    plot : bool
        Input value for ``plot``.
    real_space_pixel_size : float
        Input value for ``real_space_pixel_size``.
    **_ : Any
        Input value for ``_``.

    Returns
    -------
    result : tuple[NDArray[np.float64], NDArray[np.float64]]
        Return value produced by the function.
    """
    rng = np.random.default_rng(seed)
    rows, cols = tuple(int(v) for v in sz_array)
    diameter = float(stripe_width)
    if diameter <= 0:
        raise ValueError(f"stripe_width must be positive, got {stripe_width}")
    if skyrmion_density < 0:
        raise ValueError(f"skyrmion_density must be non-negative, got {skyrmion_density}")
    if ellipticity[0] <= 0 or ellipticity[1] <= 0 or ellipticity[0] > ellipticity[1]:
        raise ValueError(
            "ellipticity must be ordered positive bounds, "
            f"got {ellipticity}"
        )

    core_area = np.pi * (diameter / 2.0) ** 2
    target_n = max(1, int(np.round(skyrmion_density * rows * cols / core_area)))
    if target_n <= 0:
        pattern = np.ones((rows, cols), dtype=float)
        return pattern, np.empty((0, 2), dtype=float)

    candidate_count = max(1000, 50 * target_n)
    if placement_center is None:
        center_y = 0.5 * (rows - 1)
        center_x = 0.5 * (cols - 1)
    else:
        center_y, center_x = (float(placement_center[0]), float(placement_center[1]))
    center_candidate = np.array([[center_y, center_x]])

    if placement_radius is None:
        sites_arr = np.column_stack(
            [
                rng.uniform(0, rows, size=candidate_count),
                rng.uniform(0, cols, size=candidate_count),
            ]
        )
    else:
        radius = max(0.0, float(placement_radius))
        angles = rng.uniform(0, 2 * np.pi, size=candidate_count)
        radii = radius * np.sqrt(rng.uniform(0, 1, size=candidate_count))
        sites_arr = np.column_stack(
            [
                center_y + radii * np.sin(angles),
                center_x + radii * np.cos(angles),
            ]
        )
        inside = (
            (sites_arr[:, 0] >= 0)
            & (sites_arr[:, 0] < rows)
            & (sites_arr[:, 1] >= 0)
            & (sites_arr[:, 1] < cols)
        )
        sites_arr = sites_arr[inside]
    sites_arr = np.vstack([center_candidate, sites_arr])
    diameter_half_range = min(max(0.0, float(diameter_spread)), 0.2 * diameter)
    accepted: list[dict[str, float | NDArray[np.float64]]] = []
    candidate_order = np.concatenate(
        ([0], 1 + rng.permutation(len(sites_arr) - 1))
    )
    for candidate_index in candidate_order:
        if len(accepted) >= target_n:
            break
        y_center, x_center = sites_arr[candidate_index]
        local_diameter = diameter
        if diameter_half_range > 0:
            local_diameter = rng.uniform(
                diameter - diameter_half_range,
                diameter + diameter_half_range,
            )
            local_diameter = max(1.0, local_diameter)
        local_ellipticity = rng.uniform(float(ellipticity[0]), float(ellipticity[1]))
        radius = 0.5 * local_diameter
        radius_y = radius * np.sqrt(local_ellipticity)
        radius_x = radius / np.sqrt(local_ellipticity)
        roughness_scale = 1.0 + 2.0 * max(0.0, float(roughness))
        bounding_radius = max(radius_y, radius_x) * roughness_scale
        if any(
            np.hypot(y_center - item["y"], x_center - item["x"])
            < bounding_radius + item["bounding_radius"]
            for item in accepted
        ):
            continue

        accepted.append(
            {
                "y": float(y_center),
                "x": float(x_center),
                "radius_y": float(radius_y),
                "radius_x": float(radius_x),
                "angle": float(rng.uniform(0, np.pi)),
                "bounding_radius": float(bounding_radius),
            }
        )

    sites_arr = np.asarray(
        [[item["y"], item["x"]] for item in accepted], dtype=float
    )

    core = np.zeros((rows, cols), dtype=float)
    for item in accepted:
        mask = _rough_ellipse_mask(
            item["radius_y"],
            item["radius_x"],
            angle=item["angle"],
            roughness=roughness,
            roughness_modes=roughness_modes,
            rng=rng,
        )
        half_y = mask.shape[0] // 2
        half_x = mask.shape[1] // 2
        cy = int(np.round(item["y"]))
        cx = int(np.round(item["x"]))
        y0 = max(0, cy - half_y)
        y1 = min(rows, cy - half_y + mask.shape[0])
        x0 = max(0, cx - half_x)
        x1 = min(cols, cx - half_x + mask.shape[1])
        my0 = y0 - (cy - half_y)
        my1 = my0 + (y1 - y0)
        mx0 = x0 - (cx - half_x)
        mx1 = mx0 + (x1 - x0)
        core[y0:y1, x0:x1] = np.maximum(core[y0:y1, x0:x1], mask[my0:my1, mx0:mx1])

    if sigma is not None and sigma > 0:
        core = gaussian_filter(core, sigma)
        max_val = np.max(core)
        if max_val > 0:
            core = core / max_val

    pattern = 1.0 - 2.0 * np.clip(core, 0.0, 1.0)

    if plot:
        extent_real = None
        if real_space_pixel_size != 1:
            sample_y = (np.arange(rows) - rows / 2) * real_space_pixel_size
            sample_x = (np.arange(cols) - cols / 2) * real_space_pixel_size
            extent_real = 1e6 * np.array(
                [sample_x[0], sample_x[-1], sample_y[0], sample_y[-1]]
            )
        plt.figure(figsize=(5, 5))
        plt.imshow(pattern, cmap="gray", vmin=-1, vmax=1, extent=extent_real)
        plt.title("Disordered skyrmion lattice")

    return pattern, sites_arr


def create_saturated_pattern(
    sz_array: list[int] | tuple[int, int],
    saturation: float | int | str = 1,
    stripe_width: float | None = None,
    sigma: float | None = None,
    plot: bool = False,
    real_space_pixel_size: float = 1,
    **_,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Create a uniformly saturated magnetic state with ``mz = +1`` or ``-1``.

    ``stripe_width`` and ``sigma`` are accepted for compatibility with sweep
    code that uses a common magnetic-pattern parameter dictionary.

    Parameters
    ----------
    sz_array : list[int] | tuple[int, int]
        Input value for ``sz_array``.
    saturation : float | int | str
        Input value for ``saturation``.
    stripe_width : float | None
        Input value for ``stripe_width``.
    sigma : float | None
        Input value for ``sigma``.
    plot : bool
        Input value for ``plot``.
    real_space_pixel_size : float
        Input value for ``real_space_pixel_size``.
    **_ : Any
        Input value for ``_``.

    Returns
    -------
    result : tuple[NDArray[np.float64], NDArray[np.float64]]
        Return value produced by the function.
    """
    rows, cols = tuple(int(v) for v in sz_array)
    if isinstance(saturation, str):
        saturation_value = -1.0 if saturation.strip().startswith("-") else 1.0
    else:
        saturation_value = 1.0 if float(saturation) >= 0 else -1.0

    pattern = np.full((rows, cols), saturation_value, dtype=float)

    if plot:
        extent_real = None
        if real_space_pixel_size != 1:
            sample_y = (np.arange(rows) - rows / 2) * real_space_pixel_size
            sample_x = (np.arange(cols) - cols / 2) * real_space_pixel_size
            extent_real = 1e6 * np.array(
                [sample_x[0], sample_x[-1], sample_y[0], sample_y[-1]]
            )
        plt.figure(figsize=(5, 5))
        plt.imshow(pattern, cmap="gray", vmin=-1, vmax=1, extent=extent_real)
        plt.title(f"Saturated state mz={saturation_value:+.0f}")

    return pattern, np.empty((0, 2), dtype=float)


def create_wavy_stripe_pattern(
    sz_array: list[int] | tuple[int, int],
    stripe_width: float,
    angle_stripes: float,
    sigma: float | None = None,
    plot: bool = False,
    real_space_pixel_size: float = 1,
    waviness_amplitude: float = 0.0,
    waviness_scale: float = 10.0,
    seed: int | None = None,
    coordinate_offset: tuple[float, float] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Create an ordered stripe-domain pattern with optional smooth waviness.

    Parameters
    ----------
    sz_array : list[int] or tuple[int, int]
        Output array shape ``[rows, cols]`` in pixels.
    stripe_width : float
        Width of a single stripe in pixels.
    angle_stripes : float
        Stripe orientation angle in radians, measured counter-clockwise from
        the +x direction.
    sigma : float or None, optional
        Standard deviation of the Gaussian smoothing filter applied to the
        final binary stripe pattern. No smoothing when ``None``.
        Default is ``None``.
    plot : bool, optional
        If ``True``, plot the stripe pattern. Default is ``False``.
    real_space_pixel_size : float, optional
        Physical size of one pixel in metres. Used only for plotting extents.
        Default is ``1``.
    waviness_amplitude : float, optional
        Amplitude of the smooth positional disorder added to the stripe
        coordinate, in pixels. Pass ``0`` for perfectly straight stripes.
        Default is ``0.0``.
    waviness_scale : float, optional
        Standard deviation of the Gaussian filter used to smooth the random
        noise field that bends the stripes. Larger values produce broader,
        gentler undulations. Default is ``10.0``.
    seed : int or None, optional
        Random seed for reproducible waviness. Default is ``None``.
    coordinate_offset : tuple of float or None, optional
        Offset ``(y, x)`` in pixels added to the centered coordinate grid.
        This is useful when generating a cropped region of a larger stripe
        pattern while keeping the stripe phase aligned to the full sample.
        Default is ``None``.

    Returns
    -------
    pattern : ndarray of shape ``sz_array``
        Stripe pattern with values in ``[-1, 1]``.
    stripe_centers : ndarray
        1-D array of stripe-centre positions along the direction normal to
        the stripes, in pixel units.
    """
    rows, cols = sz_array
    rng = np.random.default_rng(seed)

    offset_y, offset_x = coordinate_offset or (0.0, 0.0)
    y = np.arange(rows) - rows / 2 + offset_y
    x = np.arange(cols) - cols / 2 + offset_x
    yy, xx = np.meshgrid(y, x, indexing="ij")

    # Coordinate normal to the stripe direction
    normal_coord = xx * np.cos(angle_stripes) + yy * np.sin(angle_stripes)

    # Add smooth positional disorder to make stripes gently wavy
    if waviness_amplitude > 0:
        noise = rng.standard_normal((rows, cols))
        noise = gaussian_filter(noise, waviness_scale)
        max_abs = np.max(np.abs(noise))
        if max_abs > 0:
            noise = noise / max_abs
        normal_coord = normal_coord + waviness_amplitude * noise

    stripe_index = np.floor((normal_coord + stripe_width) / stripe_width).astype(int)
    pattern = np.where(stripe_index % 2 == 0, 1.0, -1.0)

    if sigma is not None:
        pattern = gaussian_filter(pattern, sigma)
        max_val = np.max(np.abs(pattern))
        if max_val > 0:
            pattern = pattern / max_val

    min_n = normal_coord.min()
    max_n = normal_coord.max()
    stripe_centers = np.arange(min_n, max_n + stripe_width, 2 * stripe_width)

    if plot:
        if real_space_pixel_size != 1:
            sample_y = (np.arange(rows) - rows / 2) * real_space_pixel_size
            sample_x = (np.arange(cols) - cols / 2) * real_space_pixel_size
            extent_real = 1e6 * np.array(
                [sample_x[0], sample_x[-1], sample_y[0], sample_y[-1]]
            )
            xlabel = "x in µm"
            ylabel = "y in µm"
        else:
            extent_real = None
            xlabel = "x in px"
            ylabel = "y in px"

        plt.figure(figsize=(5, 5))
        plt.imshow(pattern, cmap="gray", vmin=-1, vmax=1, extent=extent_real)
        plt.title("Stripe pattern")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

    return pattern, stripe_centers


def _estimate_labyrinth_stripe_width_fft(
    pattern: NDArray[np.float64],
) -> tuple[float, float]:
    """Estimate labyrinth stripe width and full repeat period in pixels.

    The dominant FFT peak gives the wavelength of a full +/- domain repeat. A
    single stripe is one half of that repeat, matching the ``stripe_width``
    convention used by the stripe generators.

    Parameters
    ----------
    pattern : NDArray[np.float64]
        Input value for ``pattern``.

    Returns
    -------
    result : tuple[float, float]
        Return value produced by the function.
    """
    field = np.asarray(pattern, dtype=float)
    field = field - np.mean(field)
    power = np.abs(np.fft.fftshift(np.fft.fft2(field))) ** 2
    rows, cols = field.shape
    cy = rows // 2
    cx = cols // 2

    yy = np.arange(rows)[:, None] - cy
    xx = np.arange(cols)[None, :] - cx
    radius = np.sqrt(yy**2 + xx**2)
    valid = radius >= 1
    if not np.any(valid):
        period = float(min(rows, cols))
        return period / 2.0, period

    peak_index = np.argmax(power[valid])
    peak_radius = radius[valid][peak_index]
    if peak_radius <= 0:
        period = float(min(rows, cols))
        return period / 2.0, period
    period = float(min(rows, cols) / peak_radius)
    return period / 2.0, period


def _center_crop(pattern: NDArray[np.float64], shape: tuple[int, int]) -> NDArray[np.float64]:
    """Return a centered crop with ``shape``.

    The labyrinth generator intentionally does not tile images: the generated
    pattern is not periodic, so tiling can create artificial horizontal or
    vertical domain boundaries.

    Parameters
    ----------
    pattern : NDArray[np.float64]
        Input value for ``pattern``.
    shape : tuple[int, int]
        Input value for ``shape``.

    Returns
    -------
    result : NDArray[np.float64]
        Return value produced by the function.
    """
    rows, cols = shape
    if pattern.shape[0] < rows or pattern.shape[1] < cols:
        raise ValueError(
            "Cannot crop labyrinth pattern because the scaled generated image "
            f"has shape {pattern.shape}, smaller than requested shape {shape}."
        )
    y0 = (pattern.shape[0] - rows) // 2
    x0 = (pattern.shape[1] - cols) // 2
    return pattern[y0 : y0 + rows, x0 : x0 + cols]


def _center_crop_or_pad(
    pattern: NDArray[np.float64],
    shape: tuple[int, int],
    pad_mode: str = "edge",
    constant_values: float = 1.0,
) -> NDArray[np.float64]:
    """Return a centered array with ``shape``, padding if needed.

    Parameters
    ----------
    pattern : NDArray[np.float64]
        Input value for ``pattern``.
    shape : tuple[int, int]
        Input value for ``shape``.
    pad_mode : str
        Input value for ``pad_mode``.
    constant_values : float
        Input value for ``constant_values``.

    Returns
    -------
    result : NDArray[np.float64]
        Return value produced by the function.
    """
    rows, cols = shape
    src_rows, src_cols = pattern.shape

    if src_rows > rows:
        y0 = (src_rows - rows) // 2
        pattern = pattern[y0 : y0 + rows, :]
    if src_cols > cols:
        x0 = (src_cols - cols) // 2
        pattern = pattern[:, x0 : x0 + cols]

    pad_y = max(0, rows - pattern.shape[0])
    pad_x = max(0, cols - pattern.shape[1])
    if pad_y == 0 and pad_x == 0:
        return pattern

    pad_width = (
        (pad_y // 2, pad_y - pad_y // 2),
        (pad_x // 2, pad_x - pad_x // 2),
    )
    if pad_mode == "constant":
        return np.pad(
            pattern,
            pad_width,
            mode=pad_mode,
            constant_values=constant_values,
        )
    return np.pad(pattern, pad_width, mode=pad_mode)


def create_image_pattern(
    sz_array: list[int] | tuple[int, int],
    image_pixel_size: float,
    image_path: str | None = None,
    image_array: NDArray[np.float64] | None = None,
    sigma: float | None = None,
    threshold: float = 0.5,
    invert: bool = False,
    pad_mode: str = "edge",
    constant_domain: float = 1.0,
    real_space_pixel_size: float = 1,
    **_,
) -> tuple[NDArray[np.float64], dict]:
    """Create a magnetic domain pattern from an experimental binary image.

    The input image is interpreted at ``image_pixel_size`` and resampled to the
    simulation ``real_space_pixel_size``. It is binarized before resampling so
    the domain topology comes from the experimental reconstruction, while
    ``sigma`` controls the simulated domain-wall width afterward.

    Parameters
    ----------
    sz_array : list[int] | tuple[int, int]
        Input value for ``sz_array``.
    image_pixel_size : float
        Input value for ``image_pixel_size``.
    image_path : str | None
        Input value for ``image_path``.
    image_array : NDArray[np.float64] | None
        Input value for ``image_array``.
    sigma : float | None
        Input value for ``sigma``.
    threshold : float
        Input value for ``threshold``.
    invert : bool
        Input value for ``invert``.
    pad_mode : str
        Input value for ``pad_mode``.
    constant_domain : float
        Input value for ``constant_domain``.
    real_space_pixel_size : float
        Input value for ``real_space_pixel_size``.
    **_ : Any
        Input value for ``_``.

    Returns
    -------
    result : tuple[NDArray[np.float64], dict]
        Return value produced by the function.
    """
    rows, cols = tuple(sz_array)
    if image_array is None:
        if image_path is None:
            raise ValueError("image_pattern requires either image_path or image_array.")
        image = plt.imread(str(image_path))
    else:
        image = np.asarray(image_array)

    if image.ndim == 3:
        image = image[..., :3].mean(axis=-1)
    if image.ndim != 2:
        raise ValueError(
            "image pattern must be 2-D after grayscale conversion, got "
            f"{image.shape}"
        )

    image = np.asarray(image, dtype=float)
    finite = np.isfinite(image)
    if not np.any(finite):
        raise ValueError("image pattern contains no finite pixels.")
    fill_value = float(np.nanmedian(image[finite]))
    image = np.where(finite, image, fill_value)

    image_min = float(np.min(image))
    image_max = float(np.max(image))
    if image_max > image_min:
        image = (image - image_min) / (image_max - image_min)

    domains = image >= float(threshold)
    if invert:
        domains = ~domains
    pattern = np.where(domains, 1.0, -1.0)

    image_pixel_size = float(image_pixel_size)
    real_space_pixel_size = float(real_space_pixel_size)
    if image_pixel_size <= 0 or real_space_pixel_size <= 0:
        raise ValueError(
            "image_pixel_size and real_space_pixel_size must be positive, got "
            f"{image_pixel_size} and {real_space_pixel_size}."
        )
    scale = image_pixel_size / real_space_pixel_size
    if scale != 1.0:
        pattern = zoom(pattern, scale, order=0)
        if pattern.size == 0:
            raise ValueError("rescaled image pattern is empty.")

    pattern = _center_crop_or_pad(
        pattern,
        (rows, cols),
        pad_mode=str(pad_mode),
        constant_values=float(constant_domain),
    )

    if sigma is not None and sigma > 0:
        pattern = gaussian_filter(pattern, float(sigma))
        max_val = np.max(np.abs(pattern))
        if max_val > 0:
            pattern = pattern / max_val

    meta = {
        "source": "image_array" if image_array is not None else str(image_path),
        "image_pixel_size_m": image_pixel_size,
        "simulation_pixel_size_m": real_space_pixel_size,
        "rescale_factor": scale,
        "threshold": float(threshold),
        "invert": bool(invert),
        "pad_mode": str(pad_mode),
        "sigma_px": 0.0 if sigma is None else float(sigma),
    }
    return pattern, meta


def create_binary_labyrinth_pattern(
    sz_array: list[int] | tuple[int, int],
    stripe_width: float,
    sigma: float | None = None,
    domain_conversion: str = "soft",
    softness: float = 1.0,
    auto_size: bool = True,
    crop_margin: float | None = None,
    min_auto_size: int = 32,
    auto_size_overshoot: float = 1.35,
    plot: bool = False,
    real_space_pixel_size: float = 1,
    batch: int = 1,
    H: int = 100,
    W: int = 100,
    n_steps: int = 50,
    region: str | None = "custom",
    use_gpu: bool = False,
    seed: int | None = None,
    k0: float = 1.0,
    eps: float = 0.0,
    noise_amp: float = 0.0,
    **generator_overrides,
) -> tuple[NDArray[np.float64], dict]:
    """Create binary labyrinth domains rescaled to a requested stripe width.

    The fast binary generator first creates a continuous labyrinth field. Its
    stripe width is estimated as half of the dominant FFT period, then the
    continuous field is linearly rescaled so that measured width matches
    ``stripe_width`` in pixels. If needed, the base labyrinth image is
    regenerated at a larger size so that the rescaled field can be cropped
    without tiling artifacts. With ``auto_size=True``, the generated base image
    can also be smaller than the requested ``H``/``W`` when large target stripes
    imply strong upscaling. The rescaled field is then converted to the final
    domain contrast. By default this conversion is soft, using a tanh mapping
    after the optional Gaussian blur, which avoids interpolation and
    thresholding artifacts for small stripes. Set ``domain_conversion="hard"``
    to recover exact +/-1 binarisation.

    Parameters
    ----------
    sz_array : list[int] | tuple[int, int]
        Input value for ``sz_array``.
    stripe_width : float
        Input value for ``stripe_width``.
    sigma : float | None
        Input value for ``sigma``.
    domain_conversion : str
        Input value for ``domain_conversion``.
    softness : float
        Input value for ``softness``.
    auto_size : bool
        Input value for ``auto_size``.
    crop_margin : float | None
        Input value for ``crop_margin``.
    min_auto_size : int
        Input value for ``min_auto_size``.
    auto_size_overshoot : float
        Input value for ``auto_size_overshoot``.
    plot : bool
        Input value for ``plot``.
    real_space_pixel_size : float
        Input value for ``real_space_pixel_size``.
    batch : int
        Input value for ``batch``.
    H : int
        Input value for ``H``.
    W : int
        Input value for ``W``.
    n_steps : int
        Input value for ``n_steps``.
    region : str | None
        Input value for ``region``.
    use_gpu : bool
        Input value for ``use_gpu``.
    seed : int | None
        Input value for ``seed``.
    k0 : float
        Input value for ``k0``.
    eps : float
        Input value for ``eps``.
    noise_amp : float
        Input value for ``noise_amp``.
    **generator_overrides : Any
        Input value for ``generator_overrides``.

    Returns
    -------
    result : tuple[NDArray[np.float64], dict]
        Return value produced by the function.
    """
    generator_overrides.pop("coordinate_offset", None)
    rows, cols = tuple(sz_array)
    if stripe_width <= 0:
        raise ValueError(f"stripe_width must be positive, got {stripe_width}")

    requested_H = int(H)
    requested_W = int(W)
    min_auto_size = max(8, int(min_auto_size))
    auto_size_overshoot = max(1.0, float(auto_size_overshoot))
    margin = crop_margin
    if margin is None:
        margin = max(8.0, 2.0 * float(stripe_width))
        if sigma is not None:
            margin = max(margin, 4.0 * float(sigma))
    required_rows = rows + 2 * int(np.ceil(margin))
    required_cols = cols + 2 * int(np.ceil(margin))

    region_key = region[0] if isinstance(region, list) and region else region
    effective_k0 = 0.7 if region_key in (None, "labyrinth", "stripes") else float(k0)
    estimated_source_width = np.pi / effective_k0 if effective_k0 > 0 else 4.0
    estimated_scale = max(float(stripe_width) / estimated_source_width, 1e-3)
    if auto_size:
        current_H = max(
            min_auto_size,
            int(np.ceil(required_rows / estimated_scale * 1.1)),
        )
        current_W = max(
            min_auto_size,
            int(np.ceil(required_cols / estimated_scale * 1.1)),
        )
    else:
        current_H = requested_H
        current_W = requested_W

    meta = {}
    base = None
    measured_width = 0.0
    measured_period = 0.0
    scale = 1.0
    resized_after_measurement = False
    for _ in range(6):
        _, _, continuous, meta = generate_binary(
            batch=batch,
            H=current_H,
            W=current_W,
            n_steps=n_steps,
            region=region,
            use_gpu=use_gpu,
            seed=seed,
            k0=k0,
            eps=eps,
            noise_amp=noise_amp,
            **generator_overrides,
        )
        base = np.asarray(continuous[0], dtype=float)
        measured_width, measured_period = _estimate_labyrinth_stripe_width_fft(base)
        scale = stripe_width / measured_width if measured_width > 0 else 1.0
        scale = max(scale, 1e-3)
        scaled_rows = int(np.round(base.shape[0] * scale))
        scaled_cols = int(np.round(base.shape[1] * scale))
        if not auto_size:
            break

        next_H = max(
            min_auto_size,
            int(np.ceil(required_rows / scale * 1.1)),
        )
        next_W = max(
            min_auto_size,
            int(np.ceil(required_cols / scale * 1.1)),
        )
        large_enough = scaled_rows >= required_rows and scaled_cols >= required_cols
        oversized = (
            current_H > max(min_auto_size, int(np.ceil(next_H * auto_size_overshoot)))
            or current_W > max(min_auto_size, int(np.ceil(next_W * auto_size_overshoot)))
        )
        if large_enough and (not oversized or resized_after_measurement):
            break

        if large_enough and oversized:
            current_H = next_H
            current_W = next_W
            resized_after_measurement = True
        else:
            current_H = max(current_H + 1, next_H)
            current_W = max(current_W + 1, next_W)

    if base is None:
        raise RuntimeError("Labyrinth generator did not return a pattern.")

    scaled = zoom(base, scale, order=1)
    if scaled.size == 0:
        scaled = base
    pattern = _center_crop(scaled, (rows, cols))
    threshold = float(np.median(pattern))
    pattern = pattern - threshold

    if sigma is not None:
        pattern = gaussian_filter(pattern, sigma)

    conversion = str(domain_conversion).lower()
    if conversion == "hard":
        pattern = np.where(pattern >= 0, 1.0, -1.0)
    elif conversion == "soft":
        if softness <= 0:
            raise ValueError(f"softness must be positive, got {softness}")
        contrast_scale = float(np.std(pattern))
        if contrast_scale > 0:
            pattern = np.tanh(pattern / (softness * contrast_scale))
        else:
            pattern = np.zeros_like(pattern)
    else:
        raise ValueError(
            "domain_conversion must be 'soft' or 'hard', "
            f"got {domain_conversion!r}"
        )

    max_val = np.max(np.abs(pattern))
    if max_val > 0:
        pattern = pattern / max_val

    meta = dict(meta)
    meta.update(
        {
            "measured_stripe_width_px": measured_width,
            "measured_period_px": measured_period,
            "target_stripe_width_px": stripe_width,
            "rescale_factor": scale,
            "requested_H": requested_H,
            "requested_W": requested_W,
            "generated_H": current_H,
            "generated_W": current_W,
            "scaled_H": scaled.shape[0],
            "scaled_W": scaled.shape[1],
            "auto_size": bool(auto_size),
            "min_auto_size": min_auto_size,
            "auto_size_overshoot": auto_size_overshoot,
            "estimated_source_stripe_width_px": estimated_source_width,
            "crop_margin_px": margin,
            "binarization_threshold": threshold,
            "domain_conversion": conversion,
            "softness": softness,
            "k0": k0,
            "eps": eps,
            "noise_amp": noise_amp,
            "region": str(region),
            "seed": -1 if seed is None else seed,
        }
    )

    if plot:
        plt.figure(figsize=(5, 5))
        plt.imshow(pattern, cmap="gray", vmin=-1, vmax=1)
        plt.title("Binary labyrinth pattern")

    return pattern, meta


def create_stripe_pattern(
    sz_array: list[int] | tuple[int, int],
    stripe_width: float,
    angle_stripes: float,
    sigma: float | None = None,
    plot: bool = False,
    real_space_pixel_size: float = 1,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Create an ordered stripe-domain pattern.

    Parameters
    ----------
    sz_array : list[int] or tuple[int, int]
        Output array shape ``[rows, cols]`` in pixels.
    stripe_width : float
        Width of a single stripe in pixels.
    angle_stripes : float
        Stripe orientation angle in radians, measured counter-clockwise from
        the +x direction.
    sigma : float or None, optional
        Standard deviation of the Gaussian smoothing filter. No smoothing when
        ``None``. Default is ``None``.
    plot : bool, optional
        If ``True``, plot the stripe pattern. Default is ``False``.
    real_space_pixel_size : float, optional
        Physical size of one pixel in metres. Used only for plotting extents.
        Default is ``1``.

    Returns
    -------
    pattern : ndarray of shape ``sz_array``
        Stripe pattern with values in ``[-1, 1]``.
    stripe_centers : ndarray
        1-D array of stripe-center positions along the direction normal to the
        stripes, in pixel units.
    """
    rows, cols = sz_array

    y = np.arange(rows) - rows / 2
    x = np.arange(cols) - cols / 2
    yy, xx = np.meshgrid(y, x, indexing="ij")

    # Coordinate normal to the stripe direction
    normal_coord = xx * np.cos(angle_stripes) + yy * np.sin(angle_stripes)

    # Alternating +/- 1 stripes with period 2 * stripe_width
    stripe_index = np.floor((normal_coord + stripe_width) / stripe_width).astype(int)
    pattern = np.where(stripe_index % 2 == 0, 1.0, -1.0)

    if sigma is not None:
        pattern = gaussian_filter(pattern, sigma)
        max_val = np.max(np.abs(pattern))
        if max_val > 0:
            pattern = pattern / max_val

    min_n = normal_coord.min()
    max_n = normal_coord.max()
    stripe_centers = np.arange(min_n, max_n + stripe_width, 2 * stripe_width)

    if plot:
        if real_space_pixel_size != 1:
            sample_y = (np.arange(rows) - rows / 2) * real_space_pixel_size
            sample_x = (np.arange(cols) - cols / 2) * real_space_pixel_size
            extent_real = 1e6 * np.array(
                [sample_x[0], sample_x[-1], sample_y[0], sample_y[-1]]
            )
            xlabel = "x in µm"
            ylabel = "y in µm"
        else:
            extent_real = None
            xlabel = "x in px"
            ylabel = "y in px"

        plt.figure(figsize=(5, 5))
        plt.imshow(pattern, cmap="gray", vmin=-1, vmax=1, extent=extent_real)
        plt.title("Stripe pattern")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

    return pattern, stripe_centers

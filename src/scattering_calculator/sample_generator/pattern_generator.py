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


def _estimate_binary_period_fft(pattern: NDArray[np.float64]) -> float:
    """Estimate the dominant binary-domain period in pixels from the FFT peak."""
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
        return float(min(rows, cols))

    peak_index = np.argmax(power[valid])
    peak_radius = radius[valid][peak_index]
    if peak_radius <= 0:
        return float(min(rows, cols))
    return float(min(rows, cols) / peak_radius)


def _tile_center_crop(pattern: NDArray[np.float64], shape: tuple[int, int]) -> NDArray[np.float64]:
    """Tile *pattern* if needed and return a centered crop with ``shape``."""
    rows, cols = shape
    reps_y = int(np.ceil(rows / pattern.shape[0])) + 1
    reps_x = int(np.ceil(cols / pattern.shape[1])) + 1
    tiled = np.tile(pattern, (reps_y, reps_x))
    y0 = max(0, (tiled.shape[0] - rows) // 2)
    x0 = max(0, (tiled.shape[1] - cols) // 2)
    return tiled[y0 : y0 + rows, x0 : x0 + cols]


def create_binary_labyrinth_pattern(
    sz_array: list[int] | tuple[int, int],
    stripe_width: float,
    sigma: float | None = None,
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
    dominant period is estimated from the FFT, then the continuous field is
    linearly rescaled so the period matches ``stripe_width`` in pixels. The
    rescaled field is tiled/cropped to ``sz_array``, binarised, and finally
    blurred with ``sigma`` using the same convention as wavy stripes.
    """
    generator_overrides.pop("coordinate_offset", None)
    rows, cols = tuple(sz_array)
    _, binary, continuous, meta = generate_binary(
        batch=batch,
        H=H,
        W=W,
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

    measured_width = _estimate_binary_period_fft(base)
    if stripe_width <= 0:
        raise ValueError(f"stripe_width must be positive, got {stripe_width}")
    scale = stripe_width / measured_width if measured_width > 0 else 1.0
    scale = max(scale, 1e-3)
    scaled = zoom(base, scale, order=1)
    if scaled.size == 0:
        scaled = base
    pattern = _tile_center_crop(scaled, (rows, cols))
    pattern = np.where(pattern >= 0, 1.0, -1.0)

    if sigma is not None:
        pattern = gaussian_filter(pattern, sigma)
        max_val = np.max(np.abs(pattern))
        if max_val > 0:
            pattern = pattern / max_val

    meta = dict(meta)
    meta.update(
        {
            "measured_stripe_width_px": measured_width,
            "target_stripe_width_px": stripe_width,
            "rescale_factor": scale,
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

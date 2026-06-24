"""Sampling helpers for free-space propagation."""

from __future__ import annotations

import math

import numpy as np
import scipy as scp


def propagation_sampling_limit(shape, pixel_size, wavelength):
    """Return the fixed-grid propagation distance limit.

    The transfer-function/angular-spectrum step keeps the same transverse
    sampling before and after propagation. The multislice sampling criterion
    used here is ``z_max = Nt * dx**2 / wavelength``, where ``Nt`` is the
    smaller transverse pixel count.
    """
    ny, nx = int(shape[0]), int(shape[1])
    pixel_size = float(pixel_size)
    wavelength = float(wavelength)
    if ny <= 0 or nx <= 0:
        raise ValueError(f"shape must have positive transverse dimensions, got {shape!r}")
    if not np.isfinite(pixel_size) or pixel_size <= 0:
        raise ValueError(f"pixel_size must be finite and positive, got {pixel_size}")
    if not np.isfinite(wavelength) or wavelength <= 0:
        raise ValueError(f"wavelength must be finite and positive, got {wavelength}")
    return min(ny, nx) * pixel_size**2 / wavelength


def required_fixed_grid_substeps(shape, pixel_size, wavelength, dz):
    """Return the number of fixed-grid substeps needed for ``dz``."""
    dz = float(dz)
    if not np.isfinite(dz):
        raise ValueError(f"dz must be finite, got {dz}")
    if dz == 0:
        return 1
    limit = propagation_sampling_limit(shape, pixel_size, wavelength)
    if limit <= 0:
        return 1
    return max(1, int(math.ceil(abs(dz) / limit)))


def fresnel_single_fft(field, wavelength, dz, pixel_size, axes=(0, 1)):
    """Propagate with the one-FFT Fresnel transform for long distances.

    This implements

    ``psi_out(u_x lambda z, u_y lambda z) =
    F[psi_in(x, y) exp(-i pi (x^2 + y^2)/(lambda z))]
    * i/(lambda z) * exp[-i pi lambda z (u_x^2 + u_y^2)]``.

    The output grid has the same number of pixels as the input, but a different
    physical sampling ``lambda * |z| / (N * dx)``. Callers that need to apply a
    following material slice on the original grid must resample explicitly or
    use fixed-grid substepping instead.
    """
    field = np.asarray(field, dtype=complex)
    wavelength = float(wavelength)
    dz = float(dz)
    pixel_size = float(pixel_size)
    if not np.isfinite(wavelength) or wavelength <= 0:
        raise ValueError(f"wavelength must be finite and positive, got {wavelength}")
    if not np.isfinite(pixel_size) or pixel_size <= 0:
        raise ValueError(f"pixel_size must be finite and positive, got {pixel_size}")
    if not np.isfinite(dz):
        raise ValueError(f"dz must be finite, got {dz}")
    if dz == 0:
        return field.copy(), (pixel_size, pixel_size)

    y_axis, x_axis = axes
    ny = field.shape[y_axis]
    nx = field.shape[x_axis]
    y = (np.arange(ny, dtype=float) - ny // 2) * pixel_size
    x = (np.arange(nx, dtype=float) - nx // 2) * pixel_size
    yy, xx = np.meshgrid(y, x, indexing="ij")
    input_chirp = np.exp(-1j * np.pi * (xx**2 + yy**2) / (wavelength * dz))

    reshape = [1] * field.ndim
    reshape[y_axis] = ny
    reshape[x_axis] = nx
    chirped = field * input_chirp.reshape(reshape)

    spectrum = scp.fft.fftshift(
        scp.fft.fft2(
            scp.fft.ifftshift(chirped, axes=axes),
            axes=axes,
            workers=-1,
        ),
        axes=axes,
    )

    ux = scp.fft.fftshift(scp.fft.fftfreq(nx, d=pixel_size))
    uy = scp.fft.fftshift(scp.fft.fftfreq(ny, d=pixel_size))
    uy_grid, ux_grid = np.meshgrid(uy, ux, indexing="ij")
    output_chirp = (
        1j
        / (wavelength * dz)
        * np.exp(-1j * np.pi * wavelength * dz * (ux_grid**2 + uy_grid**2))
        * pixel_size**2
    )
    output = spectrum * output_chirp.reshape(reshape)
    output_pixel_size = (
        wavelength * abs(dz) / (ny * pixel_size),
        wavelength * abs(dz) / (nx * pixel_size),
    )
    return output, output_pixel_size

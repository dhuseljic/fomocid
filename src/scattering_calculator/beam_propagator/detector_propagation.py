"""Finite-distance scalar diffraction on independently sampled parallel planes.

Rayleigh--Sommerfeld I is evaluated by midpoint quadrature, with zero field
outside the supplied source window. No distance or angle expansion is made.
The exp(-ikr) convention matches the existing multislice angular spectrum.
This is the field-boundary (RS I) operator, not RS II, which requires the
normal derivative of the input field. Paganin (2006), p. 24, Eq. (1.64),
gives RS I; p. 25 states its angular-spectrum equivalence. See docs/light_propagation_modes.md
for the supplied excerpt's scope and the convention conversion.
"""

from __future__ import annotations

import numpy as np


class RayleighSommerfeldPropagator:
    """Direct, bounded-memory propagation and its Euclidean discrete adjoint.

    Coordinates and wavelength are in metres. Fields have leading axes (y, x)
    and optional trailing channels, propagated independently. Source coordinates
    are ``(index - shape/2) * pixel_size``, as in the illumination model.
    The kernel includes the source pixel area; outputs are field amplitudes,
    not pixel-integrated counts. Runtime is O(source pixels * detector pixels).
    ``max_kernel_elements`` bounds each pairwise work array, not total memory.
    """

    def __init__(self, source_shape, pixel_size, wavelength, distance,
                 detector_x, detector_y, *, max_kernel_elements=262144):
        self.source_shape = tuple(source_shape)
        if (len(self.source_shape) != 2
                or any(int(n) != n or n <= 0 for n in self.source_shape)):
            raise ValueError("source_shape must contain two positive integers")
        self.source_shape = tuple(map(int, self.source_shape))
        for name, value in (("pixel_size", pixel_size), ("wavelength", wavelength),
                            ("distance", distance)):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if int(max_kernel_elements) != max_kernel_elements or max_kernel_elements < 1:
            raise ValueError("max_kernel_elements must be a positive integer")
        self.pixel_size = float(pixel_size)
        self.k = 2 * np.pi / wavelength
        self.distance = float(distance)
        dx, dy = np.broadcast_arrays(np.asarray(detector_x, dtype=float),
                                      np.asarray(detector_y, dtype=float))
        if dx.ndim != 2 or dx.size == 0 or not (np.isfinite(dx).all() and np.isfinite(dy).all()):
            raise ValueError("detector coordinates must broadcast to a finite nonempty 2-D grid")
        self.detector_shape = dx.shape
        self.detector_x = dx.ravel().copy()
        self.detector_y = dy.ravel().copy()
        sy, sx = np.indices(self.source_shape, dtype=float)
        self.source_x = ((sx - self.source_shape[1] / 2) * pixel_size).ravel()
        self.source_y = ((sy - self.source_shape[0] / 2) * pixel_size).ravel()
        self.block_size = max(1, int(np.sqrt(max_kernel_elements)))

    def _blocks(self, source_indices=None):
        source_x = self.source_x if source_indices is None else self.source_x[source_indices]
        source_y = self.source_y if source_indices is None else self.source_y[source_indices]
        for d in range(0, self.detector_x.size, self.block_size):
            ds = slice(d, d + self.block_size)
            for s in range(0, source_x.size, self.block_size):
                ss = slice(s, s + self.block_size)
                transverse2 = (
                    (self.detector_x[ds, None] - source_x[None, ss]) ** 2
                    + (self.detector_y[ds, None] - source_y[None, ss]) ** 2
                )
                r = np.sqrt(transverse2 + self.distance**2)
                # Stable r-z avoids cancellation for long X-ray distances.
                excess = transverse2 / (r + self.distance)
                phase = np.exp(-1j * self.k * excess) * np.exp(-1j * self.k * self.distance)
                # Paganin (1.64) differentiates at the source: d/dz_source
                # = -d/d(distance). Conjugate his exp(+ikr) convention.
                # Full -d/dz [exp(-ikr)/(2*pi*r)] times source area.
                kernel = (self.pixel_size**2 * self.distance / (2 * np.pi)
                          * phase * (1 + 1j * self.k * r) / r**3)
                yield ds, ss, kernel

    def forward(self, field):
        """Propagate source amplitudes, including the full RS obliquity term."""
        field = np.asarray(field, dtype=complex)
        if field.shape[:2] != self.source_shape:
            raise ValueError("field shape does not match source_shape")
        channels = field.shape[2:]
        source = field.reshape(self.source_x.size, -1)
        # Opaque aperture pixels contribute exactly zero. Retain all nonzero
        # amplitudes without a threshold, preserving the linear operator.
        active = np.flatnonzero(np.any(source != 0, axis=1))
        source = source[active]
        result = np.zeros((self.detector_x.size, source.shape[1]), dtype=complex)
        for ds, ss, kernel in self._blocks(active):
            result[ds] += kernel @ source[ss]
        return result.reshape(self.detector_shape + channels)

    def adjoint(self, field):
        """Apply the conjugate transpose; this is not an inverse propagator."""
        field = np.asarray(field, dtype=complex)
        if field.shape[:2] != self.detector_shape:
            raise ValueError("field shape does not match detector_shape")
        channels = field.shape[2:]
        detector = field.reshape(self.detector_x.size, -1)
        result = np.zeros((self.source_x.size, detector.shape[1]), dtype=complex)
        for ds, ss, kernel in self._blocks():
            result[ss] += kernel.conj().T @ detector[ds]
        return result.reshape(self.source_shape + channels)

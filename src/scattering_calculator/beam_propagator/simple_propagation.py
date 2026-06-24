"""Scalar light-matter interaction and Fourier propagation helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy as scp
from scipy.ndimage import find_objects, label

from scattering_calculator.experimental_conditions import light_beam
from scattering_calculator.beam_propagator.free_space_sampling import (
    fresnel_single_fft,
    required_fixed_grid_substeps,
)
from scattering_calculator.utils import image_transformator


@dataclass(frozen=True)
class CompactScalarRefractiveIndexStack:
    """Layer stack represented by constant scalar indices plus ROI patches."""

    shape: tuple[int, int, int]
    base_index: np.ndarray
    patches: tuple[tuple[tuple[tuple[slice, slice], np.ndarray], ...], ...]
    aperture_support_regions: tuple[tuple[slice, slice], ...] | None = None

    @property
    def ndim(self):
        return 3

    @property
    def dtype(self):
        return self.base_index.dtype

    def __len__(self):
        return self.shape[0]

    def materialize_layer(self, layer_idx):
        _, ny, nx = self.shape
        out = np.full((ny, nx), self.base_index[layer_idx], dtype=self.dtype)
        for region, index_patch in self.patches[layer_idx]:
            out[region] = index_patch
        return out

    def materialize(self):
        return np.stack([self.materialize_layer(iz) for iz in range(len(self))])


@dataclass(frozen=True)
class LazyScalarRefractiveIndexStack:
    """Layer stack that computes scalar ROI patches only when requested."""

    shape: tuple[int, int, int]
    base_index: np.ndarray
    n_circ: np.ndarray
    n_lin: np.ndarray
    circular_coeff: complex
    linear_coeff: complex
    mask: np.ndarray
    magnetization: np.ndarray
    aperture_support: np.ndarray
    support_objects: tuple[tuple[slice, slice], ...]
    aperture_support_regions: tuple[tuple[slice, slice], ...] | None = None
    tol: float = 1e-14

    @property
    def ndim(self):
        return 3

    @property
    def dtype(self):
        return self.base_index.dtype

    def __len__(self):
        return self.shape[0]

    def layer_patches(self, layer_idx):
        patches = []
        has_circular = abs(self.circular_coeff * self.n_circ[layer_idx]) > self.tol
        has_linear = abs(self.linear_coeff * self.n_lin[layer_idx]) > self.tol

        for region in self.support_objects:
            region_support = self.aperture_support[region]
            region_mask = self.mask[(layer_idx, *region)]
            needs_patch = np.any(
                region_support & (np.abs(region_mask - 1.0) > self.tol)
            )
            if has_circular or has_linear:
                active = region_support & (region_mask > self.tol)
                needs_patch = needs_patch or np.any(active)
            if not needs_patch:
                continue

            index_patch = np.full(
                region_mask.shape, self.base_index[layer_idx], dtype=self.dtype
            )
            active = region_support & (region_mask > self.tol)
            layer_region = (layer_idx, *region)
            if has_circular and np.any(active):
                index_patch[active] += (
                    self.circular_coeff
                    * self.magnetization[layer_region + (2,)][active]
                    * self.n_circ[layer_idx]
                )
            if has_linear and np.any(active):
                mx = self.magnetization[layer_region + (0,)][active]
                my = self.magnetization[layer_region + (1,)][active]
                dxy = np.abs(mx) ** 2 - np.abs(my) ** 2
                index_patch[active] += self.linear_coeff * dxy * self.n_lin[layer_idx]

            index_patch = index_patch * region_mask + (1.0 - region_mask)
            patches.append((region, index_patch))
        return tuple(patches)

    def materialize_layer(self, layer_idx):
        _, ny, nx = self.shape
        out = np.full((ny, nx), self.base_index[layer_idx], dtype=self.dtype)
        for region, index_patch in self.layer_patches(layer_idx):
            out[region] = index_patch
        return out

    def materialize(self):
        return np.stack([self.materialize_layer(iz) for iz in range(len(self))])


class scalar_wavefronts:
    """Propagate a scalar wavefield through a complex refractive-index stack.

    This is the fast scalar counterpart to the Jones propagator. The preferred
    path consumes ``refractive_index_stack``: either a dense complex
    ``(Nz, Ny, Nx)`` array or a compact stack with constant per-layer indices
    plus aperture ROI patches. That stack is built directly from database
    channels ``[n_total, n_circ, n_lin]``. A dielectric-tensor stack can still
    be supplied for low-level Jones-comparison tests, in which case it is
    projected onto the configured polarization. If that polarization is an
    eigenvector of every local interaction matrix, the scalar result matches
    Jones propagation without carrying both Jones components.
    """

    def __init__(
        self,
        beam_parameters,
        layer_thicknesses,
        real_space_pixel_size,
        E_in,
        eps_stack=None,
        refractive_index_stack=None,
        aperture_support_regions=None,
        propagate=False,
        propagation_padding_px=0,
        propagation_padding_mode="edge",
        propagation_absorber_width_px=0,
        propagation_absorber_strength=0.0,
        propagation_absorber_profile="cosine",
        multislice_propagation_roi=False,
        multislice_propagation_roi_padding_px=0,
        multislice_propagation_roi_merge_overlaps=True,
        scalar_apply_zero_order_phase=True,
        farfield_oversampling=1,
        farfield_background_scalar=None,
    ):
        self.polarization = light_beam.polarization_vector(beam_parameters.pol)
        self.E_in = self._as_scalar_field(E_in, self.polarization)
        self.aperture_support_regions = aperture_support_regions
        self.farfield_oversampling = max(1, int(farfield_oversampling))
        self.exit_wave = self.propagate_scalar_multislice(
            E_in=self.E_in,
            eps_stack=eps_stack,
            refractive_index_stack=refractive_index_stack,
            wavelength=beam_parameters.wavelength,
            thicknesses=layer_thicknesses,
            pixel_size=real_space_pixel_size,
            polarization=self.polarization,
            propagate=propagate,
            propagation_padding_px=propagation_padding_px,
            propagation_padding_mode=propagation_padding_mode,
            propagation_absorber_width_px=propagation_absorber_width_px,
            propagation_absorber_strength=propagation_absorber_strength,
            propagation_absorber_profile=propagation_absorber_profile,
            multislice_propagation_roi=multislice_propagation_roi,
            multislice_propagation_roi_padding_px=(
                multislice_propagation_roi_padding_px
            ),
            multislice_propagation_roi_merge_overlaps=(
                multislice_propagation_roi_merge_overlaps
            ),
            scalar_apply_zero_order_phase=scalar_apply_zero_order_phase,
        )
        self.exit_wave_for_farfield = self._build_farfield_exit_wave(
            self.exit_wave,
            farfield_background_scalar,
            self.farfield_oversampling,
        )
        self.detector_wave = self.fraunhofer_propagation_scalar(
            self.exit_wave_for_farfield
        )
        self.hologram = np.abs(self.detector_wave) ** 2

    @staticmethod
    def _as_scalar_field(E_in, polarization):
        E_in = np.asarray(E_in, dtype=complex)
        if E_in.ndim == 2:
            return E_in
        if E_in.ndim == 3 and E_in.shape[-1] == 2:
            return np.einsum("...s,s->...", E_in, np.conjugate(polarization))
        raise ValueError("E_in must have shape (Ny, Nx) or (Ny, Nx, 2).")

    @staticmethod
    def _build_farfield_exit_wave(exit_wave, background_scalar, oversampling):
        oversampling = max(1, int(oversampling))
        if oversampling == 1:
            return exit_wave
        if background_scalar is None:
            raise ValueError(
                "farfield_background_scalar is required when farfield_oversampling > 1."
            )

        ny, nx = exit_wave.shape
        expected_shape = (ny * oversampling, nx * oversampling)
        extended = np.array(background_scalar, dtype=complex, copy=True)
        if extended.shape != expected_shape:
            raise ValueError(
                "farfield_background_scalar must have shape "
                f"{expected_shape}, got {extended.shape}."
            )

        y0 = (extended.shape[0] - ny) // 2
        x0 = (extended.shape[1] - nx) // 2
        extended[y0 : y0 + ny, x0 : x0 + nx] = exit_wave
        return extended

    @staticmethod
    def _is_compact_eps_stack(eps_stack):
        return (
            hasattr(eps_stack, "base_diagonal")
            and hasattr(eps_stack, "patches")
            and hasattr(eps_stack, "shape")
        )

    @staticmethod
    def _is_compact_index_stack(refractive_index_stack):
        return (
            hasattr(refractive_index_stack, "base_index")
            and (
                hasattr(refractive_index_stack, "patches")
                or hasattr(refractive_index_stack, "layer_patches")
            )
            and hasattr(refractive_index_stack, "shape")
        )

    def propagate_scalar_multislice(
        self,
        E_in,
        wavelength,
        thicknesses,
        pixel_size,
        polarization,
        eps_stack=None,
        refractive_index_stack=None,
        propagate=False,
        propagation_padding_px=0,
        propagation_padding_mode="edge",
        propagation_absorber_width_px=0,
        propagation_absorber_strength=0.0,
        propagation_absorber_profile="cosine",
        multislice_propagation_roi=False,
        multislice_propagation_roi_padding_px=0,
        multislice_propagation_roi_merge_overlaps=True,
        scalar_apply_zero_order_phase=True,
    ):
        E_in = np.asarray(E_in, dtype=complex)
        if E_in.ndim != 2:
            raise ValueError("E_in must have shape (Ny, Nx).")

        if refractive_index_stack is None and eps_stack is None:
            raise ValueError("Either refractive_index_stack or eps_stack is required.")

        use_index_stack = refractive_index_stack is not None
        if use_index_stack:
            compact_index_stack = self._is_compact_index_stack(refractive_index_stack)
            if not compact_index_stack:
                refractive_index_stack = np.asarray(
                    refractive_index_stack, dtype=complex
                )
            stack_shape = refractive_index_stack.shape
        else:
            compact_eps_stack = self._is_compact_eps_stack(eps_stack)
            if not compact_eps_stack:
                eps_stack = np.asarray(eps_stack, dtype=complex)
            stack_shape = eps_stack.shape[:3]

        if len(thicknesses) != stack_shape[0]:
            raise ValueError(
                "layer_thicknesses length must match stack first dimension."
            )
        if E_in.shape != tuple(stack_shape[1:3]):
            raise ValueError("E_in and propagation stack must have matching (Ny, Nx).")

        for iz, dz in enumerate(thicknesses):
            if use_index_stack:
                if compact_index_stack:
                    E_in = self.apply_compact_index_slice(
                        E_in, refractive_index_stack, iz, wavelength, dz
                    )
                else:
                    E_in = self.apply_index_slice(
                        E_in, refractive_index_stack[iz], wavelength, dz
                    )
            else:
                if compact_eps_stack:
                    E_in = self.apply_compact_eps_slice(
                        E_in, eps_stack, iz, wavelength, dz, polarization
                    )
                else:
                    E_in = self.apply_eps_slice(
                        E_in, eps_stack[iz], wavelength, dz, polarization
                    )

            if iz < len(thicknesses) - 1:
                if propagate:
                    if multislice_propagation_roi and self.aperture_support_regions:
                        E_in = self.propagate_free_space_scalar_roi(
                            E_in,
                            wavelength,
                            dz,
                            pixel_size,
                            self.aperture_support_regions,
                            roi_padding_px=multislice_propagation_roi_padding_px,
                            merge_overlaps=(
                                multislice_propagation_roi_merge_overlaps
                            ),
                            padding_px=propagation_padding_px,
                            padding_mode=propagation_padding_mode,
                            absorber_width_px=propagation_absorber_width_px,
                            absorber_strength=propagation_absorber_strength,
                            absorber_profile=propagation_absorber_profile,
                        )
                    else:
                        E_in = self.propagate_free_space_scalar(
                            E_in,
                            wavelength,
                            dz,
                            pixel_size,
                            padding_px=propagation_padding_px,
                            padding_mode=propagation_padding_mode,
                            absorber_width_px=propagation_absorber_width_px,
                            absorber_strength=propagation_absorber_strength,
                            absorber_profile=propagation_absorber_profile,
                        )
                elif scalar_apply_zero_order_phase:
                    k0 = 2 * np.pi / wavelength
                    E_in = E_in * np.exp(-1j * k0 * float(dz))
        return E_in

    @staticmethod
    def projected_dielectric(eps_slice, polarization):
        """Return ``p† eps p`` for a dense ``(..., 2, 2)`` tensor field."""
        eps_slice = np.asarray(eps_slice, dtype=complex)
        p = np.asarray(polarization, dtype=complex)
        return np.einsum("i,...ij,j->...", np.conjugate(p), eps_slice, p)

    def apply_index_slice(self, E, index_slice, wavelength, thickness):
        phase = -1j * (2 * np.pi / wavelength) * float(thickness)
        return E * np.exp(phase * np.asarray(index_slice, dtype=complex))

    def apply_compact_index_slice(
        self,
        E,
        refractive_index_stack,
        layer_idx,
        wavelength,
        thickness,
    ):
        phase = -1j * (2 * np.pi / wavelength) * float(thickness)
        out = E * np.exp(phase * refractive_index_stack.base_index[layer_idx])
        if hasattr(refractive_index_stack, "layer_patches"):
            layer_patches = refractive_index_stack.layer_patches(layer_idx)
        else:
            layer_patches = refractive_index_stack.patches[layer_idx]
        for region, index_patch in layer_patches:
            out[region] = self.apply_index_slice(
                E[region],
                index_patch,
                wavelength,
                thickness,
            )
        return out

    def apply_eps_slice(self, E, eps_slice, wavelength, thickness, polarization):
        phase = -1j * (2 * np.pi / wavelength) * float(thickness)
        eps_projected = self.projected_dielectric(eps_slice, polarization)
        return E * np.exp(phase * np.sqrt(eps_projected))

    def apply_compact_eps_slice(
        self,
        E,
        eps_stack,
        layer_idx,
        wavelength,
        thickness,
        polarization,
    ):
        phase = -1j * (2 * np.pi / wavelength) * float(thickness)
        p = np.asarray(polarization, dtype=complex)
        base_eps = (
            np.abs(p[0]) ** 2 * eps_stack.base_diagonal[layer_idx, 0]
            + np.abs(p[1]) ** 2 * eps_stack.base_diagonal[layer_idx, 1]
        )
        out = E * np.exp(phase * np.sqrt(base_eps))
        for region, eps_patch in eps_stack.patches[layer_idx]:
            out[region] = self.apply_eps_slice(
                E[region],
                eps_patch,
                wavelength,
                thickness,
                polarization,
            )
        return out

    @staticmethod
    def _normalize_padding_mode(mode):
        if mode in (None, "none", "None"):
            return "constant"
        return str(mode)

    @staticmethod
    def _edge_absorber(ny, nx, width_px, strength, profile="cosine"):
        width_px = max(0, int(width_px))
        if width_px == 0:
            return np.ones((ny, nx), dtype=float)
        y = np.minimum(np.arange(ny), np.arange(ny)[::-1])
        x = np.minimum(np.arange(nx), np.arange(nx)[::-1])
        dist = np.minimum(y[:, None], x[None, :])
        t = np.clip(dist / max(width_px, 1), 0.0, 1.0)
        if str(profile).lower() == "linear":
            smooth = t
        elif str(profile).lower() == "quadratic":
            smooth = t * t
        elif str(profile).lower() == "smoothstep":
            smooth = t * t * (3.0 - 2.0 * t)
        else:
            smooth = 0.5 - 0.5 * np.cos(np.pi * t)
        return np.exp(-float(strength) * (1.0 - smooth))

    @staticmethod
    def _free_space_kernel(ny, nx, wavelength, dz, pixel_size):
        fy = scp.fft.fftfreq(ny, d=pixel_size)
        fx = scp.fft.fftfreq(nx, d=pixel_size)
        fx_grid, fy_grid = np.meshgrid(fx, fy)
        k = 2 * np.pi / wavelength
        kx = 2 * np.pi * fx_grid
        ky = 2 * np.pi * fy_grid
        k_perp2 = kx**2 + ky**2
        propagating = k_perp2 <= k**2
        kz_real = np.zeros_like(k_perp2, dtype=float)
        kz_real[propagating] = np.sqrt(np.maximum(k**2 - k_perp2[propagating], 0.0))
        evanescent_decay = np.ones_like(k_perp2, dtype=float)
        if np.any(~propagating):
            alpha = np.sqrt(k_perp2[~propagating] - k**2)
            evanescent_decay[~propagating] = np.exp(-alpha * abs(dz))
        return np.exp(-1j * kz_real * dz) * evanescent_decay

    def propagate_free_space_scalar(
        self,
        E_in,
        wavelength,
        dz,
        pixel_size,
        padding_px=0,
        padding_mode="edge",
        absorber_width_px=0,
        absorber_strength=0.0,
        absorber_profile="cosine",
    ):
        E_in = np.asarray(E_in, dtype=complex)
        wavelength = float(wavelength)
        dz = float(dz)
        pixel_size = float(pixel_size)
        if dz == 0:
            return E_in.copy()

        padding_px = max(0, int(padding_px))
        padding_mode = self._normalize_padding_mode(padding_mode)
        absorber_width_px = max(0, int(absorber_width_px))
        absorber_strength = max(0.0, float(absorber_strength))
        if padding_px > 0:
            absorber_width_px = min(absorber_width_px, padding_px)
            E_work = np.pad(
                E_in,
                ((padding_px, padding_px), (padding_px, padding_px)),
                mode=padding_mode,
            )
        else:
            E_work = E_in

        if absorber_width_px > 0 and absorber_strength > 0 and padding_px == 0:
            E_work = E_work * self._edge_absorber(
                E_work.shape[0],
                E_work.shape[1],
                absorber_width_px,
                absorber_strength,
                absorber_profile,
            )

        substeps = required_fixed_grid_substeps(
            E_work.shape, pixel_size, wavelength, dz
        )
        step_dz = dz / substeps
        E_out = E_work
        for _ in range(substeps):
            H = self._free_space_kernel(*E_out.shape, wavelength, step_dz, pixel_size)
            F = scp.fft.fft2(E_out, workers=-1)
            E_out = scp.fft.ifft2(F * H, workers=-1)

        if absorber_width_px > 0 and absorber_strength > 0:
            E_out = E_out * self._edge_absorber(
                E_out.shape[0],
                E_out.shape[1],
                absorber_width_px,
                absorber_strength,
                absorber_profile,
            )
        if padding_px > 0:
            return E_out[
                padding_px : padding_px + E_in.shape[0],
                padding_px : padding_px + E_in.shape[1],
            ]
        return E_out

    @staticmethod
    def propagate_free_space_scalar_fresnel_single_fft(E_in, wavelength, dz, pixel_size):
        """Long-distance Fresnel propagation with changed output sampling.

        Returns ``(E_out, (dy_out, dx_out))``. This method implements the
        one-FFT Fresnel formula used when ``abs(dz)`` exceeds the fixed-grid
        sampling limit. The regular multislice path does not call this directly
        because material slices are sampled on the original grid; it instead
        splits long fixed-grid propagation distances into safe substeps.
        """
        return fresnel_single_fft(E_in, wavelength, dz, pixel_size, axes=(0, 1))

    def propagate_free_space_scalar_roi(
        self,
        E_in,
        wavelength,
        dz,
        pixel_size,
        aperture_support_regions,
        roi_padding_px=0,
        merge_overlaps=True,
        padding_px=0,
        padding_mode="edge",
        absorber_width_px=0,
        absorber_strength=0.0,
        absorber_profile="cosine",
    ):
        """Approximate scalar free-space propagation with FFTs in aperture ROIs.

        The full field first receives the zero-spatial-frequency phase
        ``exp(-1j * k0 * dz)``. Each local ROI crop is propagated with the same
        angular-spectrum convention used by Jones, ``exp(-1j * kz * dz)``, and
        only its deviation from the plane-wave baseline is added back.
        """
        E_in = np.asarray(E_in, dtype=complex)
        wavelength = float(wavelength)
        dz = float(dz)
        if dz == 0:
            return E_in.copy()
        if not aperture_support_regions:
            return self.propagate_free_space_scalar(
                E_in,
                wavelength,
                dz,
                pixel_size,
                padding_px=padding_px,
                padding_mode=padding_mode,
                absorber_width_px=absorber_width_px,
                absorber_strength=absorber_strength,
                absorber_profile=absorber_profile,
            )

        k0 = 2 * np.pi / wavelength
        baseline = np.asarray(E_in * np.exp(-1j * k0 * dz), dtype=complex)
        E_out = baseline.copy()

        physical_support_regions = self._merge_overlapping_regions(
            aperture_support_regions
        )
        roi_regions = self._pad_regions(
            physical_support_regions,
            E_in.shape,
            roi_padding_px,
        )
        if merge_overlaps:
            roi_regions = self._bounding_region(roi_regions)
        else:
            roi_regions = self._merge_overlapping_regions(roi_regions)

        for region in roi_regions:
            local_out = self.propagate_free_space_scalar(
                E_in[region],
                wavelength,
                dz,
                pixel_size,
                padding_px=padding_px,
                padding_mode=padding_mode,
                absorber_width_px=absorber_width_px,
                absorber_strength=absorber_strength,
                absorber_profile=absorber_profile,
            )
            correction = local_out - baseline[region]
            correction_window = self._roi_correction_window(
                correction.shape,
                E_in.shape,
                region,
                roi_padding_px,
            )
            E_out[region] += correction * correction_window
        return E_out

    @staticmethod
    def _pad_regions(regions, shape, padding_px):
        pad = max(0, int(padding_px))
        if pad == 0:
            return tuple(regions)

        ny, nx = shape
        padded = []
        for y_slice, x_slice in regions:
            y0 = max(0, int(y_slice.start or 0) - pad)
            y1 = min(ny, int(y_slice.stop) + pad)
            x0 = max(0, int(x_slice.start or 0) - pad)
            x1 = min(nx, int(x_slice.stop) + pad)
            if y1 > y0 and x1 > x0:
                padded.append((slice(y0, y1), slice(x0, x1)))
        return tuple(padded)

    @staticmethod
    def _merge_overlapping_regions(regions):
        boxes = [
            [
                int(y_slice.start or 0),
                int(y_slice.stop),
                int(x_slice.start or 0),
                int(x_slice.stop),
            ]
            for y_slice, x_slice in regions
        ]
        if not boxes:
            return ()

        def overlaps(a, b):
            return (
                a[0] < b[1]
                and b[0] < a[1]
                and a[2] < b[3]
                and b[2] < a[3]
            )

        merged = []
        for box in boxes:
            pending = box
            i = 0
            while i < len(merged):
                if overlaps(pending, merged[i]):
                    existing = merged.pop(i)
                    pending = [
                        min(pending[0], existing[0]),
                        max(pending[1], existing[1]),
                        min(pending[2], existing[2]),
                        max(pending[3], existing[3]),
                    ]
                    i = 0
                else:
                    i += 1
            merged.append(pending)

        merged.sort(key=lambda item: (item[0], item[2], item[1], item[3]))
        return tuple(
            (slice(y0, y1), slice(x0, x1)) for y0, y1, x0, x1 in merged
        )

    @staticmethod
    def _bounding_region(regions):
        boxes = [
            (
                int(y_slice.start or 0),
                int(y_slice.stop),
                int(x_slice.start or 0),
                int(x_slice.stop),
            )
            for y_slice, x_slice in regions
        ]
        if not boxes:
            return ()
        y0 = min(box[0] for box in boxes)
        y1 = max(box[1] for box in boxes)
        x0 = min(box[2] for box in boxes)
        x1 = max(box[3] for box in boxes)
        return ((slice(y0, y1), slice(x0, x1)),)

    @staticmethod
    def _roi_correction_window(local_shape, full_shape, region, taper_px):
        taper_px = max(0, int(taper_px))
        if taper_px <= 0:
            return np.ones(local_shape, dtype=float)

        ny, nx = local_shape
        full_ny, full_nx = full_shape
        y_slice, x_slice = region
        y = np.arange(ny, dtype=float)[:, None]
        x = np.arange(nx, dtype=float)[None, :]

        y_distances = []
        x_distances = []
        if int(y_slice.start or 0) > 0:
            y_distances.append(y)
        if int(y_slice.stop) < int(full_ny):
            y_distances.append(ny - 1 - y)
        if int(x_slice.start or 0) > 0:
            x_distances.append(x)
        if int(x_slice.stop) < int(full_nx):
            x_distances.append(nx - 1 - x)

        distance = np.full((ny, nx), np.inf, dtype=float)
        for item in y_distances:
            distance = np.minimum(distance, item)
        for item in x_distances:
            distance = np.minimum(distance, item)
        if np.all(np.isinf(distance)):
            return np.ones(local_shape, dtype=float)

        ramp = np.clip(distance / float(taper_px), 0.0, 1.0)
        return np.sin(0.5 * np.pi * ramp) ** 2

    @staticmethod
    def fraunhofer_propagation_scalar(E):
        return scp.fft.fftshift(scp.fft.fft2(scp.fft.ifftshift(E), workers=-1))

    def exit_wave_as_jones(self):
        return np.einsum("yx,s->yxs", self.exit_wave, self.polarization)


def scalar_intensity(E):
    """Return scalar wavefield intensity."""
    return np.abs(E) ** 2


def reconstruct(I):
    """FTH reconstruction of a scalar hologram."""
    return image_transformator.reconstruct(I)


def scalar_refractive_index_coefficients(pol):
    """Return circular and linear scalar coefficients for a polarization."""
    p = light_beam.polarization_vector(pol)
    circular_basis = np.array([[0.0, 1.0j], [-1.0j, 0.0]], dtype=complex)
    linear_basis = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    circular = np.einsum("i,ij,j->", np.conjugate(p), circular_basis, p)
    linear = np.einsum("i,ij,j->", np.conjugate(p), linear_basis, p)
    return circular, linear


def calculate_scalar_refractive_index_stack(
    sample_structure,
    pol,
    use_aperture_roi=True,
    lazy=True,
):
    """Build a compact scalar refractive-index stack from sample data.

    The database stores each layer as ``[n_total, n_circ, n_lin]``. This
    function combines those channels with the magnetization and aperture mask
    directly, without constructing dielectric tensors. Vacuum pixels receive
    ``n=1``. If ``lazy=True``, ROI patches are computed layer by layer during
    propagation instead of being precomputed for the full stack.
    """
    mask = np.asarray(sample_structure.mask, dtype=float)
    magnetization = np.asarray(sample_structure.magnetization, dtype=float)
    indices = np.asarray(sample_structure.layer_refractive_indices, dtype=complex)
    if indices.ndim != 2 or indices.shape[1] < 3:
        raise ValueError(
            "layer_refractive_indices must have shape (Nz, >=3) for scalar propagation."
        )
    if mask.shape != magnetization.shape[:3]:
        raise ValueError(
            "sample mask and magnetization must have matching (Nz, Ny, Nx)."
        )

    n0 = indices[:, 0]
    n_circ = indices[:, 1]
    n_lin = indices[:, 2]
    circular_coeff, linear_coeff = scalar_refractive_index_coefficients(pol)
    base_index = n0.astype(complex, copy=True)

    tol = 1e-14
    has_circular = np.abs(circular_coeff * n_circ) > tol
    has_linear = np.abs(linear_coeff * n_lin) > tol

    aperture_support = np.any(np.abs(1.0 - mask) > tol, axis=0)
    has_aperture_support = np.any(aperture_support)
    if not has_aperture_support:
        aperture_support = np.ones(mask.shape[1:], dtype=bool)
        support_objects = [(slice(0, mask.shape[1]), slice(0, mask.shape[2]))]
    elif use_aperture_roi:
        labels, _ = label(aperture_support)
        support_objects = [obj for obj in find_objects(labels) if obj is not None]
    else:
        support_objects = [(slice(0, mask.shape[1]), slice(0, mask.shape[2]))]

    aperture_support_regions = (
        tuple(support_objects) if use_aperture_roi and has_aperture_support else None
    )

    if lazy:
        sample_structure.scalar_refractive_index_stack = LazyScalarRefractiveIndexStack(
            shape=mask.shape,
            base_index=base_index,
            n_circ=n_circ,
            n_lin=n_lin,
            circular_coeff=circular_coeff,
            linear_coeff=linear_coeff,
            mask=mask,
            magnetization=magnetization,
            aperture_support=aperture_support,
            support_objects=tuple(support_objects),
            aperture_support_regions=aperture_support_regions,
            tol=tol,
        )
        sample_structure.aperture_support_regions = aperture_support_regions
        return sample_structure.scalar_refractive_index_stack

    patches = [[] for _ in range(mask.shape[0])]
    for layer_idx in range(mask.shape[0]):
        for region in support_objects:
            region_support = aperture_support[region]
            region_mask = mask[(layer_idx, *region)]
            needs_patch = np.any(region_support & (np.abs(region_mask - 1.0) > tol))
            if has_circular[layer_idx] or has_linear[layer_idx]:
                active = region_support & (region_mask > tol)
                needs_patch = needs_patch or np.any(active)
            if not needs_patch:
                continue

            index_patch = np.full(region_mask.shape, base_index[layer_idx], dtype=complex)
            active = region_support & (region_mask > tol)
            layer_region = (layer_idx, *region)
            if has_circular[layer_idx] and np.any(active):
                index_patch[active] += (
                    circular_coeff
                    * magnetization[layer_region + (2,)][active]
                    * n_circ[layer_idx]
                )
            if has_linear[layer_idx] and np.any(active):
                mx = magnetization[layer_region + (0,)][active]
                my = magnetization[layer_region + (1,)][active]
                dxy = np.abs(mx) ** 2 - np.abs(my) ** 2
                index_patch[active] += linear_coeff * dxy * n_lin[layer_idx]

            index_patch = index_patch * region_mask + (1.0 - region_mask)
            patches[layer_idx].append((region, index_patch))

    sample_structure.scalar_refractive_index_stack = CompactScalarRefractiveIndexStack(
        shape=mask.shape,
        base_index=base_index,
        patches=tuple(tuple(layer_patches) for layer_patches in patches),
        aperture_support_regions=aperture_support_regions,
    )
    sample_structure.aperture_support_regions = aperture_support_regions
    return sample_structure.scalar_refractive_index_stack

"""Jones-matrix propagation utilities for polarized wavefront simulations."""

import numpy as np
import scipy as scp
from scattering_calculator.utils import physics, image_transformator
from scattering_calculator.experimental_conditions import light_beam

def reconstruct(holo):
    '''
    FTH reconstruction functions

    Parameters
    ----------
    holo : (Ny, Nx) float

    Returns
    -------
    (Ny, Nx) complex
        FTH reconstruction
    '''
    return np.fft.fftshift(np.fft.fft2(np.fft.fftshift(holo)))

class wavefronts:
    _free_space_kernel_cache = {}

    def __init__(
        self,
        beam_parameters,
        eps_stack,
        layer_thicknesses,
        real_space_pixel_size,
        E_in,
        aperture_support_regions=None,
        propagate=False,
        propagation_padding_px=0,
        propagation_padding_mode="edge",
        propagation_absorber_width_px=0,
        propagation_absorber_strength=0.0,
        propagation_absorber_profile="cosine",
        multislice_propagation_roi=False,
        multislice_propagation_roi_padding_px=0,
    ):
        """Initialize a wavefronts instance.

        Parameters
        ----------
        beam_parameters : Any
            Input value for ``beam_parameters``.
        eps_stack : Any
            Input value for ``eps_stack``.
        layer_thicknesses : Any
            Input value for ``layer_thicknesses``.
        real_space_pixel_size : Any
            Input value for ``real_space_pixel_size``.
        E_in : Any
            Input value for ``E_in``.
        aperture_support_regions : Any
            Input value for ``aperture_support_regions``.
        propagate : Any
            Input value for ``propagate``.
        propagation_padding_px : Any
            Input value for ``propagation_padding_px``.
        propagation_padding_mode : Any
            Input value for ``propagation_padding_mode``.
        propagation_absorber_width_px : Any
            Input value for ``propagation_absorber_width_px``.
        propagation_absorber_strength : Any
            Input value for ``propagation_absorber_strength``.
        propagation_absorber_profile : Any
            Input value for ``propagation_absorber_profile``.
        multislice_propagation_roi : Any
            Input value for ``multislice_propagation_roi``.
        multislice_propagation_roi_padding_px : Any
            Input value for ``multislice_propagation_roi_padding_px``.

        Returns
        -------
        None
            The function completes in place.
        """
        self.E_in=E_in
        self.aperture_support_regions = aperture_support_regions
        self.propagation_padding_px = max(0, int(propagation_padding_px))
        self.propagation_padding_mode = str(propagation_padding_mode)
        self.propagation_absorber_width_px = max(0, int(propagation_absorber_width_px))
        self.propagation_absorber_strength = max(0.0, float(propagation_absorber_strength))
        self.propagation_absorber_profile = str(propagation_absorber_profile)
        self.multislice_propagation_roi = bool(multislice_propagation_roi)
        self.multislice_propagation_roi_padding_px = max(
            0, int(multislice_propagation_roi_padding_px)
        )
        self.exit_wave = self.propagate_jones_multislice(
            E_in=self.E_in,
            eps_stack=eps_stack,
            wavelength=beam_parameters.wavelength,
            thicknesses=layer_thicknesses,
            pixel_size=real_space_pixel_size,
            propagate=propagate,
            propagation_padding_px=self.propagation_padding_px,
            propagation_padding_mode=self.propagation_padding_mode,
            propagation_absorber_width_px=self.propagation_absorber_width_px,
            propagation_absorber_strength=self.propagation_absorber_strength,
            propagation_absorber_profile=self.propagation_absorber_profile,
            multislice_propagation_roi=self.multislice_propagation_roi,
            multislice_propagation_roi_padding_px=(
                self.multislice_propagation_roi_padding_px
            ),
        )
        self.detector_wave = image_transformator.Fraunhofer_propagation_jones(self.exit_wave)
        self.hologram = E_I(self.detector_wave)


    # ============================================================
    # Multislice propagation through stack of dielectric tensor images
    # ============================================================

    def propagate_jones_multislice(
        self,
        E_in,
        eps_stack,
        wavelength,
        thicknesses,
        pixel_size,
        propagate=True,
        propagation_padding_px=0,
        propagation_padding_mode="edge",
        propagation_absorber_width_px=0,
        propagation_absorber_strength=0.0,
        propagation_absorber_profile="cosine",
        multislice_propagation_roi=False,
        multislice_propagation_roi_padding_px=0,
    ):
        """
        Multislice propagation through a dielectric tensor stack.

        Parameters
        ----------
        illumination : (Ny, Nx, 2) complex
        eps_stack : (Nz, Ny, Nx, 2, 2) complex
        wavelength : float
        thicknesses : list of float
            Thicknesses of each slice
        pixel_size : float

        Returns
        -------
        E_in : (Ny, Nx, 2) complex
            Output field after all slices
        """
        E_in = np.asarray(E_in, dtype=complex)
        compact_eps_stack = self._is_compact_eps_stack(eps_stack)
        if not compact_eps_stack:
            eps_stack = np.asarray(eps_stack, dtype=complex)

        if E_in.ndim != 3 or E_in.shape[-1] != 2:
            raise ValueError("E_in must have shape (Ny, Nx, 2)")
        if eps_stack.ndim != 5 or eps_stack.shape[-2:] != (2, 2):
            raise ValueError("eps_stack must have shape (Nz, Ny, Nx, 2, 2)")
        if E_in.shape[:2] != eps_stack.shape[1:3]:
            raise ValueError("E_in and eps_stack must have matching (Ny, Nx)")

        Nz = eps_stack.shape[0]

        for iz in range(Nz):
            #print(iz)
            dz = thicknesses[iz]
            # Local Jones interaction
            if compact_eps_stack:
                E_in = self.apply_compact_eps_slice(E_in, eps_stack, iz, wavelength, dz)
            else:
                eps_slice = eps_stack[iz]
                E_in = self.propagate_jones_single_slice(
                    E_in,
                    eps_slice,
                    wavelength,
                    dz,
                    aperture_support_regions=self.aperture_support_regions,
                )

            # Free-space propagation between slices
            if propagate:
                if iz < Nz - 1:
                    if multislice_propagation_roi and self.aperture_support_regions:
                        E_in = self.propagate_free_space_jones_roi(
                            E_in,
                            wavelength,
                            dz,
                            pixel_size,
                            self.aperture_support_regions,
                            roi_padding_px=multislice_propagation_roi_padding_px,
                            padding_px=propagation_padding_px,
                            padding_mode=propagation_padding_mode,
                            absorber_width_px=propagation_absorber_width_px,
                            absorber_strength=propagation_absorber_strength,
                            absorber_profile=propagation_absorber_profile,
                        )
                    else:
                        E_in = self.propagate_free_space_jones(
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

        return E_in

    @staticmethod
    def _is_compact_eps_stack(eps_stack):
        """Handle the internal is compact eps stack operation.

        Parameters
        ----------
        eps_stack : Any
            Input value for ``eps_stack``.

        Returns
        -------
        result : Any
            Return value produced by the function.
        """
        return (
            hasattr(eps_stack, "base_diagonal")
            and hasattr(eps_stack, "patches")
            and hasattr(eps_stack, "shape")
        )


    # ============================================================
    # Single-slice propagation through dielectric tensor image
    # ============================================================

    def propagate_jones_single_slice(
        self,
        E,
        eps_slice,
        wavelength,
        thickness,
        aperture_support_regions=None,
    ):
        """
        Propagate a coherent Jones wavefield through one dielectric slice.

        Parameters
        ----------
        E : (Ny, Nx, 2) complex
            Input Jones wavefield [Ex, Ey]
        eps_slice : (Ny, Nx, 2, 2) complex
            Dielectric tensor image
        wavelength : float
        thickness : float

        Returns
        -------
        E_out : (Ny, Nx, 2) complex
        """
        eps_slice = np.asarray(eps_slice, dtype=complex)

        if E.ndim != 3 or E.shape[-1] != 2:
            raise ValueError("illumination must have shape (Ny, Nx, 2)")
        if eps_slice.ndim != 4 or eps_slice.shape[-2:] != (2, 2):
            raise ValueError("eps_slice must have shape (Ny, Nx, 2, 2)")
        if E.shape[:2] != eps_slice.shape[:2]:
            raise ValueError("E and eps_slice must have same (Ny, Nx)")

        return self.apply_eps_slice(
            E,
            eps_slice,
            wavelength,
            thickness,
            aperture_support_regions=aperture_support_regions,
        )

    def apply_eps_slice(
        self,
        E,
        eps_slice,
        wavelength,
        thickness,
        aperture_support_regions=None,
    ):
        """Apply one dielectric slice directly to a Jones wavefield.

        Most pixels in FTH simulations are diagonal after the aperture mask is
        applied. This path avoids building a full ``(Ny, Nx, 2, 2)`` Jones
        matrix field and only evaluates the 2x2 matrix function on pixels with
        non-zero off-diagonal tensor terms.

        Parameters
        ----------
        E : Any
            Input value for ``E``.
        eps_slice : Any
            Input value for ``eps_slice``.
        wavelength : Any
            Input value for ``wavelength``.
        thickness : Any
            Input value for ``thickness``.
        aperture_support_regions : Any
            Input value for ``aperture_support_regions``.

        Returns
        -------
        result : Any
            Return value produced by the function.
        """
        phase = -1j * (2 * np.pi / wavelength) * thickness
        tol = 1e-14

        a = eps_slice[..., 0, 0]
        b = eps_slice[..., 0, 1]
        c = eps_slice[..., 1, 0]
        d = eps_slice[..., 1, 1]

        if aperture_support_regions:
            a0 = a.reshape(-1)[0]
            d0 = d.reshape(-1)[0]
            E_out = np.empty_like(E, dtype=complex)
            E_out[..., 0] = np.exp(phase * np.sqrt(a0)) * E[..., 0]
            E_out[..., 1] = np.exp(phase * np.sqrt(d0)) * E[..., 1]

            for region in aperture_support_regions:
                region_key = (*region, slice(None))
                eps_region = eps_slice[region]
                E_out[region_key] = self.apply_eps_slice(
                    E[region_key],
                    eps_region,
                    wavelength,
                    thickness,
                    aperture_support_regions=None,
                )
            return E_out

        if not np.any(b) and not np.any(c):
            a0 = a.reshape(-1)[0]
            d0 = d.reshape(-1)[0]

            E_out = np.empty_like(E, dtype=complex)
            if np.all(a == a0) and np.all(d == d0):
                E_out[..., 0] = np.exp(phase * np.sqrt(a0)) * E[..., 0]
                E_out[..., 1] = np.exp(phase * np.sqrt(d0)) * E[..., 1]
                return E_out

            E_out[..., 0] = np.exp(phase * np.sqrt(a)) * E[..., 0]
            E_out[..., 1] = np.exp(phase * np.sqrt(d)) * E[..., 1]
            return E_out

        E_out = np.empty_like(E, dtype=complex)
        is_diag = (np.abs(b) < tol) & (np.abs(c) < tol)
        if np.any(is_diag):
            E_out[..., 0][is_diag] = (
                np.exp(phase * np.sqrt(a[is_diag])) * E[..., 0][is_diag]
            )
            E_out[..., 1][is_diag] = (
                np.exp(phase * np.sqrt(d[is_diag])) * E[..., 1][is_diag]
            )

        mixed = ~is_diag
        if np.any(mixed):
            aa = a[mixed]
            bb = b[mixed]
            cc = c[mixed]
            dd = d[mixed]

            tr = aa + dd
            discr = (aa - dd) ** 2 + 4 * bb * cc
            root = np.sqrt(discr)

            lam1 = 0.5 * (tr + root)
            lam2 = 0.5 * (tr - root)

            f1 = np.exp(phase * np.sqrt(lam1))
            f2 = np.exp(phase * np.sqrt(lam2))

            denom = lam1 - lam2
            regular = np.abs(denom) > tol

            alpha = np.empty_like(lam1, dtype=complex)
            beta = np.empty_like(lam1, dtype=complex)

            beta[regular] = (f1[regular] - f2[regular]) / denom[regular]
            alpha[regular] = (
                lam1[regular] * f2[regular]
                - lam2[regular] * f1[regular]
            ) / denom[regular]

            deg = ~regular
            if np.any(deg):
                lam = 0.5 * (lam1[deg] + lam2[deg])
                sqrt_lam = np.sqrt(lam)
                f = np.exp(phase * sqrt_lam)
                fp = f * phase / (2 * sqrt_lam)

                beta[deg] = fp
                alpha[deg] = f - lam * fp

            e0 = E[..., 0][mixed]
            e1 = E[..., 1][mixed]
            E_out[..., 0][mixed] = (alpha + beta * aa) * e0 + beta * bb * e1
            E_out[..., 1][mixed] = beta * cc * e0 + (alpha + beta * dd) * e1

        return E_out



    # ============================================================
    # Free-space propagation of a Jones wavefield (angular spectrum)
    # ============================================================

    def propagate_free_space_jones(
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
        """Free-space propagation of a Jones wavefield by angular spectrum.

        E_in: (Ny, Nx, 2)
        wavelength: scalar
        dz: propagation distance
        pixel_size: scalar

        returns:
            E_out: (Ny, Nx, 2)

        Parameters
        ----------
        E_in : Any
            Input value for ``E_in``.
        wavelength : Any
            Input value for ``wavelength``.
        dz : Any
            Input value for ``dz``.
        pixel_size : Any
            Input value for ``pixel_size``.
        padding_px : Any
            Input value for ``padding_px``.
        padding_mode : Any
            Input value for ``padding_mode``.
        absorber_width_px : Any
            Input value for ``absorber_width_px``.
        absorber_strength : Any
            Input value for ``absorber_strength``.
        absorber_profile : Any
            Input value for ``absorber_profile``.

        Returns
        -------
        result : Any
            Return value produced by the function.
        """
        E_in = np.asarray(E_in, dtype=complex)
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
            return E_in.copy()

        padding_px = max(0, int(padding_px))
        padding_mode = self._normalize_padding_mode(padding_mode)
        absorber_width_px = max(0, int(absorber_width_px))
        absorber_strength = max(0.0, float(absorber_strength))
        absorber_profile = str(absorber_profile).lower()
        if padding_px > 0:
            absorber_width_px = min(absorber_width_px, padding_px)
        if padding_px > 0:
            E_work = np.pad(
                E_in,
                ((padding_px, padding_px), (padding_px, padding_px), (0, 0)),
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
            )[..., None]

        Ny, Nx, _ = E_work.shape
        H = self._free_space_kernel(Ny, Nx, wavelength, dz, pixel_size)

        F = scp.fft.fft2(E_work, axes=(0, 1), workers=-1)
        F *= H[..., None]
        E_out = scp.fft.ifft2(F, axes=(0, 1), workers=-1)

        if absorber_width_px > 0 and absorber_strength > 0:
            E_out = E_out * self._edge_absorber(
                E_out.shape[0],
                E_out.shape[1],
                absorber_width_px,
                absorber_strength,
                absorber_profile,
            )[..., None]

        if padding_px > 0:
            return E_out[
                padding_px : padding_px + E_in.shape[0],
                padding_px : padding_px + E_in.shape[1],
                :,
            ]
        return E_out

    def propagate_free_space_jones_roi(
        self,
        E_in,
        wavelength,
        dz,
        pixel_size,
        aperture_support_regions,
        roi_padding_px=0,
        padding_px=0,
        padding_mode="edge",
        absorber_width_px=0,
        absorber_strength=0.0,
        absorber_profile="cosine",
    ):
        """Approximate free-space propagation with FFTs only in aperture ROIs.

        Outside the ROI boxes the field is assumed locally plane-wave-like and
        receives only the zero-spatial-frequency angular-spectrum phase.
        Padded ROI boxes that overlap are merged before propagation, so nearby
        apertures are treated as one local diffraction problem instead of
        separate crops competing in shared pixels. Each merged ROI contributes
        only its deviation from the plane-wave baseline. This is faster than a
        global FFT but intentionally approximate because true free-space
        propagation couples all pixels.

        Parameters
        ----------
        E_in : Any
            Input value for ``E_in``.
        wavelength : Any
            Input value for ``wavelength``.
        dz : Any
            Input value for ``dz``.
        pixel_size : Any
            Input value for ``pixel_size``.
        aperture_support_regions : Any
            Input value for ``aperture_support_regions``.
        roi_padding_px : Any
            Input value for ``roi_padding_px``.
        padding_px : Any
            Input value for ``padding_px``.
        padding_mode : Any
            Input value for ``padding_mode``.
        absorber_width_px : Any
            Input value for ``absorber_width_px``.
        absorber_strength : Any
            Input value for ``absorber_strength``.
        absorber_profile : Any
            Input value for ``absorber_profile``.

        Returns
        -------
        result : Any
            Return value produced by the function.
        """
        E_in = np.asarray(E_in, dtype=complex)
        wavelength = float(wavelength)
        dz = float(dz)
        if dz == 0:
            return E_in.copy()
        if not aperture_support_regions:
            return self.propagate_free_space_jones(
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

        roi_regions = self._merge_overlapping_regions(
            self._pad_regions(
                aperture_support_regions,
                E_in.shape[:2],
                roi_padding_px,
            )
        )
        for region in roi_regions:
            region_key = (*region, slice(None))
            local_out = self.propagate_free_space_jones(
                E_in[region_key],
                wavelength,
                dz,
                pixel_size,
                padding_px=padding_px,
                padding_mode=padding_mode,
                absorber_width_px=absorber_width_px,
                absorber_strength=absorber_strength,
                absorber_profile=absorber_profile,
            )
            E_out[region_key] += local_out - baseline[region_key]

        return E_out

    @staticmethod
    def _pad_regions(regions, shape, padding_px):
        """Return y/x slice regions padded and clipped to ``shape``.

        Parameters
        ----------
        regions : Any
            Input value for ``regions``.
        shape : Any
            Input value for ``shape``.
        padding_px : Any
            Input value for ``padding_px``.

        Returns
        -------
        result : Any
            Return value produced by the function.
        """
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
        """Return bounding boxes formed by merging overlapping y/x regions.

        Parameters
        ----------
        regions : Any
            Iterable of ``(y_slice, x_slice)`` regions.

        Returns
        -------
        result : tuple
            Tuple of non-overlapping ``(y_slice, x_slice)`` regions.
        """
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

    @classmethod
    def _normalize_padding_mode(cls, padding_mode):
        """Return a NumPy padding mode for free-space propagation margins.

        Parameters
        ----------
        padding_mode : Any
            Input value for ``padding_mode``.

        Returns
        -------
        result : Any
            Return value produced by the function.
        """
        mode = str(padding_mode).lower()
        aliases = {
            "zero": "constant",
            "zeros": "constant",
            "constant": "constant",
            "edge": "edge",
            "reflect": "reflect",
            "symmetric": "symmetric",
        }
        if mode not in aliases:
            raise ValueError(
                "padding_mode must be one of 'edge', 'reflect', 'symmetric', "
                f"or 'constant', got {padding_mode!r}."
            )
        return aliases[mode]

    @classmethod
    def _edge_absorber(cls, Ny, Nx, width_px, strength, profile="cosine"):
        """Return a smooth edge absorber equal to one away from the border.

        Parameters
        ----------
        Ny : Any
            Input value for ``Ny``.
        Nx : Any
            Input value for ``Nx``.
        width_px : Any
            Input value for ``width_px``.
        strength : Any
            Input value for ``strength``.
        profile : Any
            Input value for ``profile``.

        Returns
        -------
        result : Any
            Return value produced by the function.
        """
        width_px = max(0, int(width_px))
        strength = max(0.0, float(strength))
        if width_px <= 0 or strength <= 0:
            return np.ones((Ny, Nx), dtype=float)

        yy = np.minimum(np.arange(Ny), np.arange(Ny)[::-1]).astype(float)
        xx = np.minimum(np.arange(Nx), np.arange(Nx)[::-1]).astype(float)
        dist = np.minimum(yy[:, None], xx[None, :])
        ramp = np.clip(dist / float(width_px), 0.0, 1.0)
        edge_weight = cls._absorber_edge_weight(ramp, profile)
        absorber = np.ones((Ny, Nx), dtype=float)
        edge = ramp < 1.0
        absorber[edge] = np.exp(-strength * edge_weight[edge])
        return absorber

    @classmethod
    def _absorber_edge_weight(cls, ramp, profile):
        """Return a smooth 0-to-1 absorption profile from interior to edge.

        Parameters
        ----------
        ramp : Any
            Input value for ``ramp``.
        profile : Any
            Input value for ``profile``.

        Returns
        -------
        result : Any
            Return value produced by the function.
        """
        profile = str(profile).lower()
        t = 1.0 - np.clip(ramp, 0.0, 1.0)
        if profile == "linear":
            return t
        if profile == "quadratic":
            return t**2
        if profile == "cosine":
            return 0.5 * (1.0 - np.cos(np.pi * t))
        if profile == "smoothstep":
            return t * t * (3.0 - 2.0 * t)
        raise ValueError(
            "absorber_profile must be one of 'cosine', 'smoothstep', "
            f"'quadratic', or 'linear', got {profile!r}."
        )

    @classmethod
    def _free_space_kernel(cls, Ny, Nx, wavelength, dz, pixel_size):
        """Return a cached angular-spectrum propagator for one free-space step.

        Parameters
        ----------
        Ny : Any
            Input value for ``Ny``.
        Nx : Any
            Input value for ``Nx``.
        wavelength : Any
            Input value for ``wavelength``.
        dz : Any
            Input value for ``dz``.
        pixel_size : Any
            Input value for ``pixel_size``.

        Returns
        -------
        result : Any
            Return value produced by the function.
        """
        key = (int(Ny), int(Nx), float(wavelength), float(dz), float(pixel_size))
        H = cls._free_space_kernel_cache.get(key)
        if H is not None:
            return H

        k0 = 2 * np.pi / wavelength
        fx = np.fft.fftfreq(Nx, d=pixel_size)
        fy = np.fft.fftfreq(Ny, d=pixel_size)
        FX, FY = np.meshgrid(fx, fy, indexing="xy")
        k_perp2 = (2 * np.pi * FX) ** 2 + (2 * np.pi * FY) ** 2

        propagating = k_perp2 <= k0**2
        kz_real = np.zeros_like(k_perp2, dtype=float)
        kz_real[propagating] = np.sqrt(np.maximum(k0**2 - k_perp2[propagating], 0.0))
        evanescent_decay = np.ones_like(k_perp2, dtype=float)
        if np.any(~propagating):
            alpha = np.sqrt(k_perp2[~propagating] - k0**2)
            evanescent_decay[~propagating] = np.exp(-alpha * abs(dz))
        H = np.asarray(
            np.exp(-1j * kz_real * dz) * evanescent_decay,
            dtype=np.complex128,
        )

        cls._free_space_kernel_cache[key] = H
        return H

    def propagate_free_space_jones_260526(self,E_in, wavelength, dz, pixel_size):
        """Previous per-polarization free-space propagation implementation kept
        for comparison/debugging. New code should use
        ``propagate_free_space_jones``.

        Parameters
        ----------
        E_in : Any
            Input value for ``E_in``.
        wavelength : Any
            Input value for ``wavelength``.
        dz : Any
            Input value for ``dz``.
        pixel_size : Any
            Input value for ``pixel_size``.

        Returns
        -------
        result : Any
            Return value produced by the function.
        """
        E_in = np.asarray(E_in, dtype=complex)
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
            return E_in.copy()

        Ny, Nx, _ = E_in.shape
        k0 = 2 * np.pi / wavelength
        fx = np.fft.fftfreq(Nx, d=pixel_size)
        fy = np.fft.fftfreq(Ny, d=pixel_size)
        FX, FY = np.meshgrid(fx, fy, indexing="xy")
        k_perp2 = (2 * np.pi * FX) ** 2 + (2 * np.pi * FY) ** 2

        propagating = k_perp2 <= k0**2
        kz_real = np.zeros_like(k_perp2, dtype=float)
        kz_real[propagating] = np.sqrt(np.maximum(k0**2 - k_perp2[propagating], 0.0))
        evanescent_decay = np.ones_like(k_perp2, dtype=float)
        if np.any(~propagating):
            alpha = np.sqrt(k_perp2[~propagating] - k0**2)
            evanescent_decay[~propagating] = np.exp(-alpha * abs(dz))
        H = np.exp(-1j * kz_real * dz) * evanescent_decay

        E_out = np.zeros_like(E_in, dtype=complex)
        for pol in range(2):
            F = np.fft.fft2(E_in[..., pol])
            F_prop = F * H
            E_out[..., pol] = np.fft.ifft2(F_prop)

        return E_out

    def apply_compact_eps_slice(self, E, eps_stack, layer_idx, wavelength, thickness):
        """Apply one compact dielectric layer without materializing the full slice.

        Parameters
        ----------
        E : Any
            Input value for ``E``.
        eps_stack : Any
            Input value for ``eps_stack``.
        layer_idx : Any
            Input value for ``layer_idx``.
        wavelength : Any
            Input value for ``wavelength``.
        thickness : Any
            Input value for ``thickness``.

        Returns
        -------
        result : Any
            Return value produced by the function.
        """
        phase = -1j * (2 * np.pi / wavelength) * thickness
        base_a, base_d = eps_stack.base_diagonal[layer_idx]

        E_out = np.empty_like(E, dtype=complex)
        E_out[..., 0] = np.exp(phase * np.sqrt(base_a)) * E[..., 0]
        E_out[..., 1] = np.exp(phase * np.sqrt(base_d)) * E[..., 1]

        for region, eps_patch in eps_stack.patches[layer_idx]:
            region_key = (*region, slice(None))
            E_out[region_key] = self.apply_eps_slice(
                E[region_key],
                eps_patch,
                wavelength,
                thickness,
                aperture_support_regions=None,
            )
        return E_out


    # ============================================================
    # Utility: apply a Jones matrix field to a Jones wavefield
    # ============================================================
    def apply_jones_field(self,E_in, J_field):
        """E_in:   (Ny, Nx, 2)
        J_field:(Ny, Nx, 2, 2)

        returns:
            E_out: (Ny, Nx, 2)

        Parameters
        ----------
        E_in : Any
            Input value for ``E_in``.
        J_field : Any
            Input value for ``J_field``.

        Returns
        -------
        result : Any
            Return value produced by the function.
        """
        return np.einsum("yxab,yxb->yxa", J_field, E_in)


def E_j(E,pol):
    """Calculate Jones wavefield for given polarization.

    Parameters
    ----------
    E : ndarray of shape (Ny, Nx, 2)
        Input Jones wavefield.
    pol : int or str
        Polarization index (0 for Ex, 1 for Ey) or polarization type ("CR

    Returns
    -------
    result : Any
        Return value produced by the function.
    """

    return np.einsum("yxs,s->yxs", E, light_beam.polarization_vector(pol))

def E_I(E):
    """Calculate intensity of Jones wavefield for given polarization.

    Parameters
    ----------
    E : ndarray of shape (Ny, Nx, 2)
        Input Jones wavefield.

    Returns
    -------
    result : Any
        Return value produced by the function.
    """
    I= (np.sum(np.abs(E)**2, axis=(2)))
    return I

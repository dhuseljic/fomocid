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
    def __init__(
        self,
        beam_parameters,
        eps_stack,
        layer_thicknesses,
        real_space_pixel_size,
        E_in,
        aperture_support_regions=None,
        propagate=False,
    ):
        self.E_in=E_in
        self.aperture_support_regions = aperture_support_regions
        self.exit_wave = self.propagate_jones_multislice(
            E_in=self.E_in,
            eps_stack=eps_stack,
            wavelength=beam_parameters.wavelength,
            thicknesses=layer_thicknesses,
            pixel_size=real_space_pixel_size,
            propagate=propagate
        )
        self.detector_wave = image_transformator.Fraunhofer_propagation_jones(self.exit_wave)
        self.hologram = E_I(self.detector_wave)


    # ============================================================
    # Multislice propagation through stack of dielectric tensor images
    # ============================================================

    def propagate_jones_multislice(self,E_in, eps_stack, wavelength, thicknesses, pixel_size, propagate=True):
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
            eps_slice = eps_stack[iz]
            dz = thicknesses[iz]
            # Local Jones interaction
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
                    E_in = self.propagate_free_space_jones(E_in, wavelength, dz, pixel_size)

        return E_in


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

    def propagate_free_space_jones(self,E_in, wavelength, dz, pixel_size):
        """
        Free-space propagation of a Jones wavefield by angular spectrum.

        E_in: (Ny, Nx, 2)
        wavelength: scalar
        dz: propagation distance
        pixel_size: scalar

        returns:
            E_out: (Ny, Nx, 2)
        """
        E_in = np.asarray(E_in, dtype=complex)
        Ny, Nx, _ = E_in.shape

        k0 = 2 * np.pi / wavelength

        fx = np.fft.fftfreq(Nx, d=pixel_size)
        fy = np.fft.fftfreq(Ny, d=pixel_size)
        FX, FY = np.meshgrid(fx, fy, indexing="xy")

        kx = 2 * np.pi * FX
        ky = 2 * np.pi * FY

        kz = np.sqrt((k0**2 - kx**2 - ky**2) + 0j)
        H = np.exp(-1j * kz * dz)

        E_out = np.zeros_like(E_in, dtype=complex)

        for pol in range(2):
            F = np.fft.fft2(E_in[..., pol])
            F_prop = F * H
            E_out[..., pol] = np.fft.ifft2(F_prop)

        return E_out


    # ============================================================
    # Utility: apply a Jones matrix field to a Jones wavefield
    # ============================================================
    def apply_jones_field(self,E_in, J_field):
        """
        E_in:   (Ny, Nx, 2)
        J_field:(Ny, Nx, 2, 2)

        returns:
            E_out: (Ny, Nx, 2)
        """
        return np.einsum("yxab,yxb->yxa", J_field, E_in)


def E_j(E,pol):
    '''
    Calculate Jones wavefield for given polarization.
    
    Parameters
    ----------
    E : ndarray of shape (Ny, Nx, 2)
        Input Jones wavefield.
    pol : int or str
        Polarization index (0 for Ex, 1 for Ey) or polarization type ("CR'''

    return np.einsum("yxs,s->yxs", E, light_beam.polarization_vector(pol))

def E_I(E):
    '''Calculate intensity of Jones wavefield for given polarization.
    
    Parameters
    ----------
    E : ndarray of shape (Ny, Nx, 2)
        Input Jones wavefield.       
    '''
    I= (np.sum(np.abs(E)**2, axis=(2)))
    return I

"""Archived Jones-matrix propagation utilities kept for comparison."""

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
    def __init__(self, beam_parameters,eps_stack,layer_thicknesses,real_space_pixel_size,E_in, propagate=False):
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
        propagate : Any
            Input value for ``propagate``.

        Returns
        -------
        None
            The function completes in place.
        """
        self.E_in=E_in
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
            E_in = self.propagate_jones_single_slice(E_in, eps_slice, wavelength, dz)

            # Free-space propagation between slices
            if propagate:
                if iz < Nz - 1:
                    E_in = self.propagate_free_space_jones(E_in, wavelength, dz, pixel_size)

        return E_in


    # ============================================================
    # Single-slice propagation through dielectric tensor image
    # ============================================================

    def propagate_jones_single_slice(self,E, eps_slice, wavelength, thickness):
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

        return self.apply_eps_slice(E, eps_slice, wavelength, thickness)

    def apply_eps_slice(self, E, eps_slice, wavelength, thickness):
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
    # Build Jones propagator field from dielectric tensor field
    # ============================================================

    def jones_from_eps_slice(self, eps_slice, wavelength, thickness):
        """Fast vectorized Jones propagator for a field of 2x2 dielectric tensors.

        eps_slice: (Ny, Nx, 2, 2)
        returns:   (Ny, Nx, 2, 2)

        Computes:
            J = exp(-i k0 thickness sqrt(eps))
        using the 2x2 matrix-function identity:
            f(eps) = alpha I + beta eps

        Parameters
        ----------
        eps_slice : Any
            Input value for ``eps_slice``.
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
        tol = 1e-14

        a = eps_slice[..., 0, 0]
        b = eps_slice[..., 0, 1]
        c = eps_slice[..., 1, 0]
        d = eps_slice[..., 1, 1]

        # Fast path: whole slice is exactly diagonal.
        # This is cheap to test and avoids mask/gather/scatter overhead.
        if not np.any(b) and not np.any(c):
            a0 = a.reshape(-1)[0]
            d0 = d.reshape(-1)[0]

            # Fastest path: whole layer is one uniform diagonal tensor.
            if np.all(a == a0) and np.all(d == d0):
                j00 = np.exp(phase * np.sqrt(a0))
                j11 = np.exp(phase * np.sqrt(d0))

                J = np.zeros_like(eps_slice, dtype=complex)
                J[..., 0, 0] = j00
                J[..., 1, 1] = j11
                return J

            # Whole slice diagonal, but spatially varying.
            J = np.zeros_like(eps_slice, dtype=complex)
            J[..., 0, 0] = np.exp(phase * np.sqrt(a))
            J[..., 1, 1] = np.exp(phase * np.sqrt(d))
            return J


        # Fallback: your original mixed-case logic
        J = np.zeros_like(eps_slice, dtype=complex)

        is_diag = (np.abs(b) < tol) & (np.abs(c) < tol)

        if np.any(is_diag):
            J[..., 0, 0][is_diag] = np.exp(phase * np.sqrt(a[is_diag]))
            J[..., 1, 1][is_diag] = np.exp(phase * np.sqrt(d[is_diag]))

        mask = ~is_diag

        if np.any(mask):
            aa = a[mask]
            bb = b[mask]
            cc = c[mask]
            dd = d[mask]

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

            J_sub = np.empty((aa.size, 2, 2), dtype=complex)
            J_sub[:, 0, 0] = alpha + beta * aa
            J_sub[:, 0, 1] = beta * bb
            J_sub[:, 1, 0] = beta * cc
            J_sub[:, 1, 1] = alpha + beta * dd

            J[mask] = J_sub

        return J




    def jones_from_eps_slice_new(self, eps_slice, wavelength, thickness):
        """Fast vectorized Jones propagator for a field of 2x2 dielectric tensors.

        eps_slice: (Ny, Nx, 2, 2)
        returns:   (Ny, Nx, 2, 2)

        Computes:
            J = exp(-i k0 thickness sqrt(eps))
        using the 2x2 matrix-function identity:
            f(eps) = alpha I + beta eps

        Parameters
        ----------
        eps_slice : Any
            Input value for ``eps_slice``.
        wavelength : Any
            Input value for ``wavelength``.
        thickness : Any
            Input value for ``thickness``.

        Returns
        -------
        result : Any
            Return value produced by the function.
        """
        k0 = 2 * np.pi / wavelength
        a = eps_slice[..., 0, 0]
        b = eps_slice[..., 0, 1]
        c = eps_slice[..., 1, 0]
        d = eps_slice[..., 1, 1]

        # Eigenvalues of 2x2 matrix [[a,b],[c,d]]
        tr = a + d
        det_discriminant = (a - d)**2 + 4 * b * c
        root = np.sqrt(det_discriminant)

        lam1 = 0.5 * (tr + root)
        lam2 = 0.5 * (tr - root)

        f1 = np.exp(-1j * k0 * thickness * np.sqrt(lam1))
        f2 = np.exp(-1j * k0 * thickness * np.sqrt(lam2))

        denom = lam1 - lam2
        tol = 1e-14

        J = np.empty_like(eps_slice, dtype=complex)

        regular = np.abs(denom) > tol

        beta = np.empty_like(lam1, dtype=complex)
        alpha = np.empty_like(lam1, dtype=complex)

        # Non-degenerate case
        beta[regular] = (f1[regular] - f2[regular]) / denom[regular]
        alpha[regular] = (
            lam1[regular] * f2[regular]
            - lam2[regular] * f1[regular]
        ) / denom[regular]

        # Degenerate case: lam1 ~= lam2
        # f(eps) ≈ f(lam) I + f'(lam) (eps - lam I)
        # so beta = f'(lam), alpha = f(lam) - lam f'(lam)
        deg = ~regular
        if np.any(deg):
            lam = 0.5 * (lam1[deg] + lam2[deg])
            sqrt_lam = np.sqrt(lam)
            f = np.exp(-1j * k0 * thickness * sqrt_lam)

            # derivative of exp(-i k0 t sqrt(lambda))
            fp = f * (-1j * k0 * thickness) / (2 * sqrt_lam)

            beta[deg] = fp
            alpha[deg] = f - lam * fp

        # J = alpha I + beta eps
        J[..., 0, 0] = alpha + beta * a
        J[..., 0, 1] = beta * b
        J[..., 1, 0] = beta * c
        J[..., 1, 1] = alpha + beta * d

        return J


    def jones_from_eps_slice_old(self,eps_slice, wavelength, thickness):
        """eps_slice: (Ny, Nx, 2, 2)
        wavelength: scalar
        thickness: scalar

        returns:
            J_field: (Ny, Nx, 2, 2)

        Optimized: Detects isotropic pixels (diagonal with equal elements)
        and computes their Jones matrix directly without eigendecomposition.
        Uses eig for general complex matrices and avoids explicit inversion.

        Parameters
        ----------
        eps_slice : Any
            Input value for ``eps_slice``.
        wavelength : Any
            Input value for ``wavelength``.
        thickness : Any
            Input value for ``thickness``.

        Returns
        -------
        result : Any
            Return value produced by the function.
        """
        k0 = 2 * np.pi / wavelength
        Ny, Nx = eps_slice.shape[:2]

        # Extract diagonal and off-diagonal elements
        diag_00 = eps_slice[..., 0, 0]  # (Ny, Nx)
        diag_11 = eps_slice[..., 1, 1]  # (Ny, Nx)
        off_01 = eps_slice[..., 0, 1]   # (Ny, Nx)
        off_10 = eps_slice[..., 1, 0]   # (Ny, Nx)

        # Check which pixels are isotropic: diagonal with equal elements
        tol = 1e-10
        is_isotropic = (
            (np.abs(off_01) < tol) &
            (np.abs(off_10) < tol) &
            (np.abs(diag_00 - diag_11) < tol * np.abs(diag_00) + tol)
        )

        # Initialize Jones field
        J_field = np.zeros((Ny, Nx, 2, 2), dtype=complex)

        # Handle isotropic pixels directly (no eigendecomposition needed)
        if np.any(is_isotropic):
            n_iso = np.sqrt(diag_00[is_isotropic])
            phase_iso = np.exp(-1j * k0 * n_iso * thickness)

            # For isotropic: J = diag(phase, phase)
            J_field[is_isotropic, 0, 0] = phase_iso
            J_field[is_isotropic, 1, 1] = phase_iso

        # Handle anisotropic pixels with eigendecomposition
        if np.any(~is_isotropic):
            eps_aniso = eps_slice[~is_isotropic]
            N_aniso = len(eps_aniso)

            # Use eig for general complex matrices (handles non-Hermitian case with complex diagonals)
            vals, vecs = np.linalg.eig(eps_aniso)  # vals: (N_aniso, 2), vecs: (N_aniso, 2, 2)
            print(vals[0], vecs[0])

            # Refractive indices
            n = np.sqrt(vals)  # (N_aniso, 2)

            # Propagation phases
            phase = np.exp(-1j * k0 * n * thickness)  # (N_aniso, 2)

            # Compute J = V @ diag(phase) @ V^{-1} efficiently
            # Use solve instead of explicit inversion (much faster)
            eye = np.eye(2, dtype=complex)
            vecs_inv = np.linalg.solve(vecs, np.tile(eye[np.newaxis, :, :], (N_aniso, 1, 1)))

            # Efficient computation using einsum: J[i,a,b] = sum_c V[i,a,c] * phase[i,c] * V_inv[i,c,b]
            J_aniso = np.einsum('ijk,ik,ikl->ijl', vecs, phase, vecs_inv)

            # Place anisotropic results in output
            J_field[~is_isotropic] = J_aniso

        return J_field



    def jones_from_eps_slice_oldold(self,eps_slice, wavelength, thickness):
        """eps_slice: (Ny, Nx, 2, 2)
        wavelength: scalar
        thickness: scalar

        returns:
            J_field: (Ny, Nx, 2, 2)

        Parameters
        ----------
        eps_slice : Any
            Input value for ``eps_slice``.
        wavelength : Any
            Input value for ``wavelength``.
        thickness : Any
            Input value for ``thickness``.

        Returns
        -------
        result : Any
            Return value produced by the function.
        """
        k0 = 2 * np.pi / wavelength

        # Eigen-decomposition for Hermitian matrices (faster than general eig)
        vals, vecs = np.linalg.eig(eps_slice)     # vals: (Ny,Nx,2), vecs: (Ny,Nx,2,2)

        print("---",eps_slice[eps_slice.shape[0]//2,eps_slice.shape[0]//2],"\n---",vals[eps_slice.shape[0]//2,eps_slice.shape[0]//2],"\n---",vecs[eps_slice.shape[0]//2,eps_slice.shape[0]//2])
        print(".")# Refractive indices of local eigenmodes
        n = np.sqrt(vals)                         # (Ny,Nx,2)

        # Propagation phases
        phase = np.exp(-1j * k0 * n * thickness)   # (Ny,Nx,2)

        # Build diagonal matrix field
        D = np.zeros_like(eps_slice, dtype=complex)
        D[..., 0, 0] = phase[..., 0]
        D[..., 1, 1] = phase[..., 1]

        # J = V D V^{-1}
        vecs_inv = np.linalg.inv(vecs)
        J_field = vecs @ D @ vecs_inv
        return J_field


    # ============================================================
    # Free-space propagation of a Jones wavefield (angular spectrum)
    # ============================================================

    def propagate_free_space_jones(self,E_in, wavelength, dz, pixel_size):
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

        Returns
        -------
        result : Any
            Return value produced by the function.
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

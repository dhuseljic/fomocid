import numpy as np
import scipy as scp


class exit_wave:
    def __init__(self, sample,E_in, propagate=False):
        self.E = self.propagate_jones_multislice(
            E_in=E_in,
            eps_stack=np.array(sample.final_dielectric_tensor),
            wavelength=physics.photon_energy_wavelength(x_ray_energy),
            thicknesses=sample.layer_thicknesses,
            pixel_size=sample.real_space_pixel_size,
            propagate=propagate
        )




    # ============================================================
    # Multislice propagation through stack of dielectric tensor images
    # ============================================================

    def propagate_jones_multislice(E-in, eps_stack, wavelength, thicknesses, pixel_size, propagate=True):
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
        E : (Ny, Nx, 2) complex
            Output field after all slices
        """
        E = np.asarray(E_in, dtype=complex)
        eps_stack = np.asarray(eps_stack, dtype=complex)

        if E.ndim != 3 or E.shape[-1] != 2:
            raise ValueError("E_in must have shape (Ny, Nx, 2)")
        if eps_stack.ndim != 5 or eps_stack.shape[-2:] != (2, 2):
            raise ValueError("eps_stack must have shape (Nz, Ny, Nx, 2, 2)")
        if E.shape[:2] != eps_stack.shape[1:3]:
            raise ValueError("E_in and eps_stack must have matching (Ny, Nx)")

        Nz = eps_stack.shape[0]

        for iz in range(Nz):
            eps_slice = eps_stack[iz]
            dz = thicknesses[iz]
            # Local Jones interaction
            E = propagate_jones_single_slice(E, eps_slice, wavelength, dz)

            # Free-space propagation between slices
            if propagate:
                if iz < Nz - 1:
                    E = propagate_free_space_jones(E, wavelength, dz, pixel_size)

        return E




    # ============================================================
    # Single-slice propagation through dielectric tensor image
    # ============================================================

    def propagate_jones_single_slice(E, eps_slice, wavelength, thickness):
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

        J_field = jones_from_eps_slice(eps_slice, wavelength, thickness)
        E_out = apply_jones_field(E, J_field)

        return E_out


    # ============================================================
    # Build Jones propagator field from dielectric tensor field
    # ============================================================

    def jones_from_eps_slice(eps_slice, wavelength, thickness):
        """
        eps_slice: (Ny, Nx, 2, 2)
        wavelength: scalar
        thickness: scalar

        returns:
            J_field: (Ny, Nx, 2, 2)
        
        Optimized: Detects isotropic pixels (diagonal with equal elements)
        and computes their Jones matrix directly without eigendecomposition.
        """
        k0 = 2 * np.pi / wavelength
        Ny, Nx = eps_slice.shape[:2]
        
        # Extract diagonal and off-diagonal elements
        diag_00 = eps_slice[..., 0, 0]  # (Ny, Nx)
        diag_11 = eps_slice[..., 1, 1]  # (Ny, Nx)
        off_01 = eps_slice[..., 0, 1]   # (Ny, Nx)
        off_10 = eps_slice[..., 1, 0]   # (Ny, Nx)
        
        # Check which pixels are isotropic: diagonal with equal elements
        # Tolerance for floating point comparison
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
            
            # Eigen-decomposition for anisotropic pixels only
            vals, vecs = np.linalg.eig(eps_aniso)  # vals: (N_aniso, 2), vecs: (N_aniso, 2, 2)
            
            # Refractive indices
            n = np.sqrt(vals)  # (N_aniso, 2)
            
            # Propagation phases
            phase = np.exp(-1j * k0 * n * thickness)  # (N_aniso, 2)
            
            # Build diagonal phase matrices
            D = np.zeros((len(eps_aniso), 2, 2), dtype=complex)
            D[:, 0, 0] = phase[:, 0]
            D[:, 1, 1] = phase[:, 1]
            
            # J = V D V^{-1}
            vecs_inv = np.linalg.inv(vecs)
            J_aniso = vecs @ D @ vecs_inv
            
            # Place anisotropic results in output
            J_field[~is_isotropic] = J_aniso
        
        return J_field



    def jones_from_eps_slice_old(eps_slice, wavelength, thickness):
        """
        eps_slice: (Ny, Nx, 2, 2)
        wavelength: scalar
        thickness: scalar

        returns:
            J_field: (Ny, Nx, 2, 2)
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

    def propagate_free_space_jones(E_in, wavelength, dz, pixel_size):
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
        #print(kz,dz,kz*dz)
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
    def apply_jones_field(E_in, J_field):
        """
        E_in:   (Ny, Nx, 2)
        J_field:(Ny, Nx, 2, 2)

        returns:
            E_out: (Ny, Nx, 2)
        """
        return np.einsum("yxab,yxb->yxa", J_field, E_in)






def polarization_vector(pol):
    '''
    Return the Jones vector for a given polarization type.
    
    Parameters
    ----------
    pol : str or float
        Polarization type: "CR" (circular right), "CL" (circular left), "x" (linear horizontal), "y" (linear vertical), or angle in radians for linear polarization at that angle.
    
    Returns
    -------
    jones_vec : ndarray of shape (2,)
        Jones vector corresponding to the specified polarization.
    '''

    if pol=="CR":
        return np.array([1, -1j]) / np.sqrt(2)
    elif pol=="CL":
        return np.array([1, 1j]) / np.sqrt(2)
    elif pol=="x":
        return np.array([1, 0])
    elif pol=="y":
        return np.array([0, 1])
    else:
        return np.array([np.sin(pol), np.cos(pol)]) 
    
    
def scalar_to_jones(scalar_wavefield, pol):
    '''Calculate Jones wavefield for given polarization.
    
    Parameters
    ----------
    scalar_wavefield : ndarray of shape (Ny, Nx)
        Input scalar wavefield.
    pol : int or str
        Polarization index (0 for Ex, 1 for Ey) or polarization type ("CR", "CL", "x", "y", or angle in radians).        
    
    Returns
    -------
    Jones wavefield : ndarray of shape (Ny, Nx, 2)
        Jones wavefield corresponding to the specified polarization.
    '''
    return np.einsum("yx,s->yxs", scalar_wavefield, polarization_vector(pol))


def E_j(E,pol):
    '''
    Calculate Jones wavefield for given polarization.
    
    Parameters
    ----------
    E : ndarray of shape (Ny, Nx, 2)
        Input Jones wavefield.
    pol : int or str
        Polarization index (0 for Ex, 1 for Ey) or polarization type ("CR'''

    return np.einsum("yxs,s->yxs", E, polarization_vector(pol))

def E_I(E,pol):
    '''Calculate intensity of Jones wavefield for given polarization.
    
    Parameters
    ----------
    E : ndarray of shape (Ny, Nx, 2)
        Input Jones wavefield.
    pol : int
        Polarization index (0 for Ex, 1 for Ey).        
    '''
    I= (np.sum(np.abs(E)**2, axis=(2)))
    return I


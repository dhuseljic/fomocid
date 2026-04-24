import numpy as np


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
    """
    k0 = 2 * np.pi / wavelength

    # Eigen-decomposition pixel by pixel
    vals, vecs = np.linalg.eig(eps_slice)     # vals: (Ny,Nx,2), vecs: (Ny,Nx,2,2)

    # Refractive indices of local eigenmodes
    n = np.sqrt(vals)                         # (Ny,Nx,2)

    # Propagation phases
    print(k0, n[0,0], thickness)
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
# Single-slice propagation through dielectric tensor image
# ============================================================

def propagate_jones_single_slice(illumination, eps_slice, wavelength, thickness):
    """
    Propagate a coherent Jones wavefield through one dielectric slice.

    Parameters
    ----------
    illumination : (Ny, Nx, 2) complex
        Input Jones wavefield [Ex, Ey]
    eps_slice : (Ny, Nx, 2, 2) complex
        Dielectric tensor image
    wavelength : float
    thickness : float

    Returns
    -------
    E_out : (Ny, Nx, 2) complex
    J_field : (Ny, Nx, 2, 2) complex
    """
    illumination = np.asarray(illumination, dtype=complex)
    eps_slice = np.asarray(eps_slice, dtype=complex)

    if illumination.ndim != 3 or illumination.shape[-1] != 2:
        raise ValueError("illumination must have shape (Ny, Nx, 2)")
    if eps_slice.ndim != 4 or eps_slice.shape[-2:] != (2, 2):
        raise ValueError("eps_slice must have shape (Ny, Nx, 2, 2)")
    if illumination.shape[:2] != eps_slice.shape[:2]:
        raise ValueError("illumination and eps_slice must have same (Ny, Nx)")

    J_field = jones_from_eps_slice(eps_slice, wavelength, thickness)
    E_out = apply_jones_field(illumination, J_field)
    return E_out, J_field


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
# Multislice propagation through stack of dielectric tensor images
# ============================================================

def propagate_jones_multislice(illumination, eps_stack, wavelength, thicknesses, pixel_size, propagate=True):
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
    E = np.asarray(illumination, dtype=complex)
    eps_stack = np.asarray(eps_stack, dtype=complex)

    if E.ndim != 3 or E.shape[-1] != 2:
        raise ValueError("illumination must have shape (Ny, Nx, 2)")
    if eps_stack.ndim != 5 or eps_stack.shape[-2:] != (2, 2):
        raise ValueError("eps_stack must have shape (Nz, Ny, Nx, 2, 2)")
    if E.shape[:2] != eps_stack.shape[1:3]:
        raise ValueError("illumination and eps_stack must have matching (Ny, Nx)")

    Nz = eps_stack.shape[0]

    for iz in range(Nz):
        eps_slice = eps_stack[iz]
        dz = thicknesses[iz]
        # Local Jones interaction
        E, _ = propagate_jones_single_slice(E, eps_slice, wavelength, dz)

        # Free-space propagation between slices
        if propagate:
            if iz < Nz - 1:
                E = propagate_free_space_jones(E, wavelength, dz, pixel_size)

    return E




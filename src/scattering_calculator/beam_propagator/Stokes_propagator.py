"""Mueller--Stokes utilities and wavefront propagation.

The public arrays in this module use the component order ``[I, Q, U, V]``.
The sign convention is chosen so that the project's existing ``"CR"`` Jones
vector, ``[1, -1j] / sqrt(2)``, has ``V = +I``.  Stating that convention here
is important: opposite signs for circular polarization are both common in the
optics literature.

There is a subtle distinction between *polarization* coherence and *spatial*
coherence.  A four-component Stokes value describes the former at one point,
but contains no phase relation between different pixels.  Consequently a
Stokes image cannot, by itself, be propagated through an FFT without losing
the information needed to form a diffraction pattern.  ``stokes_wavefronts``
therefore retains one or more coherent two-component carriers for spatial
propagation and publishes their incoherent Stokes sum at the input, exit, and
detector planes.  A fully polarized beam needs one carrier.  A partially
polarized uniform Stokes beam is decomposed into orthogonal coherent modes of
its 2x2 coherency matrix, which is the usual way to propagate a deterministic
linear optical system while preserving the unpolarized fraction correctly.
At every plane the polarization transformation is exactly the corresponding
Mueller transformation for the deterministic dielectric tensors used by this
simulator, while leaving room for explicit depolarizing Mueller elements in
the standalone helpers below.
"""

from __future__ import annotations

import numpy as np

from scattering_calculator.beam_propagator import Jones_propagator


# Hermitian basis matrices satisfying S_i = trace(C sigma_i), where C is the
# 2x2 polarization coherency matrix.  Keeping the basis in one place prevents
# tiny convention differences from leaking into the conversion routines.
_STOKES_BASIS = np.asarray(
    [
        [[1, 0], [0, 1]],       # I: total intensity
        [[1, 0], [0, -1]],      # Q: horizontal minus vertical
        [[0, 1], [1, 0]],       # U: +45 degree linear component
        [[0, 1j], [-1j, 0]],    # V: right-circular is positive in this project
    ],
    dtype=complex,
)


def jones_to_stokes(field: np.ndarray) -> np.ndarray:
    """Convert a coherent Jones field to Stokes form.

    Parameters
    ----------
    field : ndarray
        Complex array with final dimension two.

    Returns
    -------
    ndarray
        Real array with final dimension four in ``[I, Q, U, V]`` order.
    """
    field = np.asarray(field, dtype=complex)
    if field.ndim < 1 or field.shape[-1] != 2:
        raise ValueError("Jones field must have shape (..., 2).")

    ex, ey = field[..., 0], field[..., 1]
    cross = ex * np.conjugate(ey)
    return np.stack(
        (
            np.abs(ex) ** 2 + np.abs(ey) ** 2,
            np.abs(ex) ** 2 - np.abs(ey) ** 2,
            2.0 * np.real(cross),
            2.0 * np.imag(cross),
        ),
        axis=-1,
    )


def stokes_to_coherency(stokes: np.ndarray) -> np.ndarray:
    """Return the 2x2 coherency matrix represented by a Stokes field.

    Unlike a Jones vector, a coherency matrix can represent partially
    polarized and unpolarized light.  This conversion is therefore lossless
    for every physically valid Stokes vector.
    Parameters
    ----------
    stokes : ndarray
        Real Stokes vectors with final dimension four.

    Returns
    -------
    ndarray
        Complex coherency matrices with final dimensions ``(2, 2)``.
    """
    stokes = np.asarray(stokes, dtype=float)
    if stokes.ndim < 1 or stokes.shape[-1] != 4:
        raise ValueError("Stokes field must have shape (..., 4).")
    return 0.5 * np.einsum("...i,iab->...ab", stokes, _STOKES_BASIS)


def coherency_to_stokes(coherency: np.ndarray) -> np.ndarray:
    """Convert coherency matrices to real Stokes vectors.

    Parameters
    ----------
    coherency : ndarray
        Complex array with final dimensions ``(2, 2)``.

    Returns
    -------
    ndarray
        Real Stokes vectors with final dimension four.
    """
    coherency = np.asarray(coherency, dtype=complex)
    if coherency.ndim < 2 or coherency.shape[-2:] != (2, 2):
        raise ValueError("Coherency field must have shape (..., 2, 2).")
    values = np.einsum("...ab,iba->...i", coherency, _STOKES_BASIS)
    # Physical coherency matrices produce real values.  real_if_close avoids
    # hiding a genuinely invalid matrix while removing round-off-sized parts.
    values = np.real_if_close(values, tol=1000)
    if np.iscomplexobj(values):
        raise ValueError("Coherency matrix is not Hermitian.")
    return np.asarray(values, dtype=float)


def jones_to_mueller(jones: np.ndarray) -> np.ndarray:
    """Convert Jones matrices ``(..., 2, 2)`` to Mueller matrices ``(..., 4, 4)``.

    The formula is ``M_ij = 1/2 Tr(sigma_i J sigma_j J^dagger)``.  Such a
    matrix is necessarily non-depolarizing; arbitrary depolarizing Mueller
    matrices can still be supplied directly to :func:`apply_mueller`.
    Parameters
    ----------
    jones : ndarray
        Jones matrices with final dimensions ``(2, 2)``.

    Returns
    -------
    ndarray
        Real Mueller matrices with final dimensions ``(4, 4)``.
    """
    jones = np.asarray(jones, dtype=complex)
    if jones.ndim < 2 or jones.shape[-2:] != (2, 2):
        raise ValueError("Jones matrix must have shape (..., 2, 2).")
    mueller = 0.5 * np.einsum(
        "iab,...bc,jcd,...ad->...ij",
        _STOKES_BASIS,
        jones,
        _STOKES_BASIS,
        np.conjugate(jones),
        optimize=True,
    )
    mueller = np.real_if_close(mueller, tol=1000)
    if np.iscomplexobj(mueller):
        raise ValueError("Jones-to-Mueller conversion produced non-real values.")
    return np.asarray(mueller, dtype=float)


def apply_mueller(stokes: np.ndarray, mueller: np.ndarray) -> np.ndarray:
    """Apply one Mueller matrix or a spatial Mueller field to Stokes vectors.

    Parameters
    ----------
    stokes : ndarray
        Stokes vectors with final dimension four.
    mueller : ndarray
        Mueller matrix or field with final dimensions ``(4, 4)``.

    Returns
    -------
    ndarray
        Transformed Stokes vectors.
    """
    stokes = np.asarray(stokes, dtype=float)
    mueller = np.asarray(mueller, dtype=float)
    if stokes.ndim < 1 or stokes.shape[-1] != 4:
        raise ValueError("Stokes field must have shape (..., 4).")
    if mueller.ndim < 2 or mueller.shape[-2:] != (4, 4):
        raise ValueError("Mueller matrix must have shape (..., 4, 4).")
    try:
        return np.einsum("...ij,...j->...i", mueller, stokes)
    except ValueError as exc:
        raise ValueError("Mueller and Stokes leading dimensions do not broadcast.") from exc


def validate_stokes(stokes: np.ndarray, *, atol: float = 1e-10) -> None:
    """Raise ``ValueError`` unless every Stokes vector is physically admissible.

    Parameters
    ----------
    stokes : ndarray
        Stokes vectors to validate.
    atol : float, optional
        Absolute tolerance for floating-point physicality checks.

    Returns
    -------
    None
        Validation completes without a return value.
    """
    stokes = np.asarray(stokes, dtype=float)
    if stokes.ndim < 1 or stokes.shape[-1] != 4:
        raise ValueError("Stokes field must have shape (..., 4).")
    if not np.all(np.isfinite(stokes)):
        raise ValueError("Stokes field contains non-finite values.")
    polarized_length = np.linalg.norm(stokes[..., 1:], axis=-1)
    if np.any(stokes[..., 0] < -atol) or np.any(
        polarized_length > stokes[..., 0] + atol
    ):
        raise ValueError("Unphysical Stokes vector: require I >= sqrt(Q^2+U^2+V^2).")


def _coherency_to_modes(stokes: np.ndarray, *, atol: float = 1e-12):
    """Return non-negative coherent polarization modes for one Stokes vector."""
    stokes = np.asarray(stokes, dtype=float)
    if stokes.shape != (4,):
        raise ValueError(
            "input_stokes must be one uniform Stokes vector with shape (4,)."
        )
    validate_stokes(stokes, atol=atol)
    coherency = stokes_to_coherency(stokes)
    eigenvalues, eigenvectors = np.linalg.eigh(coherency)

    modes = []
    for value, vector in zip(eigenvalues, eigenvectors.T):
        # Tiny negative eigenvalues are numerical noise near a pure state.
        if value > atol:
            modes.append((float(value), vector.astype(complex, copy=False)))
    if not modes:
        raise ValueError("input_stokes must contain non-zero intensity.")
    return modes


def _field_for_mode(template: np.ndarray, weight: float, vector: np.ndarray) -> np.ndarray:
    """Build a Jones field with ``template`` intensity and one polarization mode."""
    template = np.asarray(template, dtype=complex)
    if template.ndim < 1 or template.shape[-1] != 2:
        raise ValueError("Jones template must have shape (..., 2).")
    intensity_envelope = np.sum(np.abs(template) ** 2, axis=-1)
    amplitude = np.sqrt(np.maximum(intensity_envelope, 0.0) * weight)
    return amplitude[..., np.newaxis] * vector


class stokes_wavefronts:
    """Propagate deterministic sample response and expose Mueller--Stokes fields.

    The dielectric stack currently represents deterministic, non-depolarizing
    material response.  Its Mueller action is exactly equivalent to the Jones
    carrier used here to preserve spatial phase through free space.  Public
    ``input_stokes``, ``exit_stokes``, and ``detector_stokes`` arrays make the
    polarization state explicit, while ``hologram`` is detector-plane ``I``.
    """

    def __init__(self, *args, input_stokes=None, **kwargs):
        """Initialize a Stokes wavefront around the coherent propagator.

        Parameters
        ----------
        *args : tuple
            Positional arguments accepted by ``Jones_propagator.wavefronts``.
        input_stokes : array_like, optional
            Uniform incident Stokes vector ``[I, Q, U, V]``.  If omitted, the
            incident Jones field passed as ``E_in`` is treated as a pure state,
            preserving the original Jones-equivalent behavior.  If supplied,
            the Stokes vector may be partially polarized and is decomposed into
            incoherent coherent modes before propagation.
        **kwargs : dict
            Keyword arguments accepted by ``Jones_propagator.wavefronts``.

        Returns
        -------
        None
            Initialization completes in place.
        """
        if input_stokes is None:
            # Reuse the mature spatial propagation machinery without modifying
            # its code path.  This is deliberate: scalar and Jones behavior
            # remain byte-for-byte untouched, and Stokes adds a facade around it.
            self._coherent_carrier = Jones_propagator.wavefronts(*args, **kwargs)
            self._coherent_modes = [self._coherent_carrier]

            carrier = self._coherent_carrier
            self.input_jones = np.asarray(carrier.E_in, dtype=complex)
            self.exit_jones = carrier.exit_wave
            self.detector_jones = carrier.detector_wave
            self.exit_jones_for_farfield = carrier.exit_wave_for_farfield

            self.input_stokes = jones_to_stokes(self.input_jones)
            self.exit_stokes = jones_to_stokes(self.exit_jones)
            self.detector_stokes = jones_to_stokes(self.detector_jones)
            self.exit_stokes_for_farfield = jones_to_stokes(
                self.exit_jones_for_farfield
            )
        else:
            # A partially polarized beam has no single Jones vector.  We use
            # the spectral decomposition of its coherency matrix: each
            # eigenvector is propagated coherently, then intensities/Stokes
            # vectors are summed incoherently.  This keeps the unpolarized part
            # from acquiring artificial phase coherence.
            jones_template = np.asarray(kwargs["E_in"], dtype=complex)
            background_template = kwargs.get("farfield_background_jones")
            mode_specs = _coherency_to_modes(input_stokes)
            self._coherent_modes = []
            self._mode_weights_and_vectors = mode_specs

            self.input_stokes = 0.0
            self.exit_stokes = 0.0
            self.detector_stokes = 0.0
            self.exit_stokes_for_farfield = 0.0
            for weight, vector in mode_specs:
                mode_kwargs = dict(kwargs)
                mode_kwargs["E_in"] = _field_for_mode(jones_template, weight, vector)
                if background_template is not None:
                    mode_kwargs["farfield_background_jones"] = _field_for_mode(
                        background_template, weight, vector
                    )
                carrier = Jones_propagator.wavefronts(*args, **mode_kwargs)
                self._coherent_modes.append(carrier)
                self.input_stokes = self.input_stokes + jones_to_stokes(carrier.E_in)
                self.exit_stokes = self.exit_stokes + jones_to_stokes(
                    carrier.exit_wave
                )
                self.detector_stokes = self.detector_stokes + jones_to_stokes(
                    carrier.detector_wave
                )
                self.exit_stokes_for_farfield = (
                    self.exit_stokes_for_farfield
                    + jones_to_stokes(carrier.exit_wave_for_farfield)
                )

            # Keep a representative complex carrier for legacy phase-bearing
            # consumers.  For mixed states it is only the dominant coherent
            # mode; the physically meaningful public field is exit_stokes.
            self._coherent_carrier = self._coherent_modes[0]
            self.input_jones = np.asarray(self._coherent_carrier.E_in, dtype=complex)
            self.exit_jones = self._coherent_carrier.exit_wave
            self.detector_jones = self._coherent_carrier.detector_wave
            self.exit_jones_for_farfield = self._coherent_carrier.exit_wave_for_farfield

        # Use Stokes fields as the primary public wavefront representation.
        # Compatibility aliases for complex fields remain explicitly named.
        self.E_in = self.input_stokes
        self.exit_wave = self.exit_stokes
        self.exit_wave_for_farfield = self.exit_stokes_for_farfield
        self.detector_wave = self.detector_stokes
        self.hologram = self.detector_stokes[..., 0].copy()


# A CamelCase alias reads naturally in new code, while the lower-case class
# follows the naming convention of the two pre-existing propagators.
StokesWavefronts = stokes_wavefronts


__all__ = [
    "StokesWavefronts",
    "apply_mueller",
    "coherency_to_stokes",
    "jones_to_mueller",
    "jones_to_stokes",
    "stokes_to_coherency",
    "stokes_wavefronts",
    "validate_stokes",
]

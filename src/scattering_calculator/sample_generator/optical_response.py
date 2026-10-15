"""Optional nonmagnetic optical contributions to the existing dielectric tensor."""
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike


@dataclass
class OpticalResponseConfig:
    """Enable a material-dataset channel and specify its independent spatial order.

    Optical coefficients always come from the same material_params database as
    the isotropic and magnetic channels, at the current photon energy. No spectra
    or constants are defined here. Axis/strength specify structural orientation,
    polar-axis order or natural handedness, independently of magnetization.
    """

    material: str
    kind: Literal["natural_linear", "natural_circular", "ferroelectric"]
    axis: ArrayLike | None = (1.0, 0.0, 0.0)
    strength: ArrayLike = 1.0

    def validate(self):
        """Validate the material/channel selector without inventing optical data.

        Parameters
        ----------
        None
            Uses this spatial response configuration.

        Returns
        -------
        None
            Unknown channel names and invalid material selectors raise ValueError.
        """
        if self.kind not in ("natural_linear", "natural_circular", "ferroelectric"):
            raise ValueError(f"Unknown optical response kind: {self.kind!r}")
        if not isinstance(self.material, str) or not self.material:
            raise ValueError("Optical response material must be a layer material name")

    def spatial_tensor(self, sample_shape, layer_index):
        """Construct the dimensionless axis dyadic or isotropic gyration tensor.

        Parameters
        ----------
        sample_shape : tuple
            Untilted material-grid shape (Nz, Ny, Nx).
        layer_index : int
            Subdivided material layer to evaluate.

        Returns
        -------
        ndarray
            Weighted axis dyadic, shape (Ny, Nx, 3, 3).
        """
        nz, ny, nx = sample_shape
        strength = np.asarray(self.strength, dtype=float)
        if strength.ndim == 3:
            if strength.shape != (nz, ny, nx):
                raise ValueError(
                    "strength volume must match the subdivided sample grid"
                )
            strength = strength[layer_index]
        try:
            strength = np.broadcast_to(strength, (ny, nx))
        except ValueError as error:
            raise ValueError(
                "strength must be scalar, (Ny,Nx), or (Nz,Ny,Nx)"
            ) from error
        if not np.isfinite(strength).all():
            raise ValueError("Optical response strength must be finite")
        if self.axis is None:
            if self.kind != "natural_circular":
                raise ValueError("Linear and ferroelectric responses require an axis")
            return strength[..., None, None] * np.eye(3)
        axis = np.asarray(self.axis, dtype=float)
        if axis.ndim == 4:
            if axis.shape != (nz, ny, nx, 3):
                raise ValueError("axis volume must match the subdivided sample grid")
            axis = axis[layer_index]
        try:
            axis = np.broadcast_to(axis, (ny, nx, 3))
        except ValueError as error:
            raise ValueError("axis must be (3,), (Ny,Nx,3), or (Nz,Ny,Nx,3)") from error
        if not np.isfinite(axis).all():
            raise ValueError("Optical response axis must be finite")
        norm = np.linalg.norm(axis, axis=-1, keepdims=True)
        if np.any(norm == 0):
            raise ValueError(
                "Axis directions must be nonzero; use strength=0 for absent order"
            )
        unit = axis / norm
        return strength[..., None, None] * unit[..., :, None] * unit[..., None, :]

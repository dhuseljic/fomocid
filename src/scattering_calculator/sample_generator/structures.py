"""Sample, multilayer, aperture, and recipe data structures."""

from __future__ import annotations
import numpy as np
from numpy.typing import ArrayLike, NDArray
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.ndimage import gaussian_filter, label, find_objects
from scattering_calculator.utils.masking import circle_mask
from scattering_calculator.utils.masking import circle_mask3D

from scattering_calculator.database.database_loading import (
    material_params,
)  # noqa: E402

from pathlib import Path

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Tuple


@dataclass(frozen=True)
class CompactDielectricTensorStack:
    """Layer stack represented by constant diagonals plus aperture ROI patches."""

    shape: tuple[int, int, int, int, int]
    base_diagonal: NDArray[np.complex128]
    patches: tuple[tuple[tuple[tuple[slice, slice], NDArray[np.complex128]], ...], ...]
    aperture_support_regions: tuple[tuple[slice, slice], ...] | None = None

    @property
    def ndim(self) -> int:
        """Run the ndim operation.

        Parameters
        ----------
        None
            This function takes no explicit input parameters.

        Returns
        -------
        result : int
            Return value produced by the function.
        """
        return 5

    @property
    def dtype(self) -> np.dtype:
        """Run the dtype operation.

        Parameters
        ----------
        None
            This function takes no explicit input parameters.

        Returns
        -------
        result : np.dtype
            Return value produced by the function.
        """
        return self.base_diagonal.dtype

    def __len__(self) -> int:
        """Handle the internal len operation.

        Parameters
        ----------
        None
            This function takes no explicit input parameters.

        Returns
        -------
        result : int
            Return value produced by the function.
        """
        return self.shape[0]

    def materialize_layer(self, layer_idx: int) -> NDArray[np.complex128]:
        """Return one dense ``(Ny, Nx, 2, 2)`` dielectric tensor slice.

        Parameters
        ----------
        layer_idx : int
            Input value for ``layer_idx``.

        Returns
        -------
        result : NDArray[np.complex128]
            Return value produced by the function.
        """
        _, ny, nx, _, _ = self.shape
        out = np.zeros((ny, nx, 2, 2), dtype=self.dtype)
        out[..., 0, 0] = self.base_diagonal[layer_idx, 0]
        out[..., 1, 1] = self.base_diagonal[layer_idx, 1]
        for region, eps_patch in self.patches[layer_idx]:
            out[region] = eps_patch
        return out

    def materialize(self) -> NDArray[np.complex128]:
        """Return the full dense ``(Nz, Ny, Nx, 2, 2)`` tensor stack.

        Parameters
        ----------
        None
            This function takes no explicit input parameters.

        Returns
        -------
        result : NDArray[np.complex128]
            Return value produced by the function.
        """
        return np.stack([self.materialize_layer(iz) for iz in range(len(self))])


@dataclass(frozen=True)
class DynamicProjectedDielectricTensorStack:
    """Magnetic response stack projected onto a local beam direction at runtime."""

    eps_iso: NDArray[np.complex128]
    gyrotropic_vector: NDArray[np.complex128]
    linear_tensor: NDArray[np.complex128]
    beam_direction: NDArray[np.float64]
    aperture_support_regions: tuple[tuple[slice, slice], ...] | None = None

    @property
    def shape(self) -> tuple[int, int, int, int, int]:
        """Dense Jones tensor shape produced after projection."""
        return (*self.eps_iso.shape, 2, 2)

    @property
    def ndim(self) -> int:
        """NumPy-like ndim for compatibility with propagation checks."""
        return 5

    @property
    def dtype(self) -> np.dtype:
        """Complex dtype of projected dielectric tensors."""
        return self.eps_iso.dtype

    @property
    def base_diagonal(self) -> NDArray[np.complex128]:
        """Nominal diagonal response used for background estimates."""
        base = np.empty((self.eps_iso.shape[0], 2), dtype=self.eps_iso.dtype)
        for layer_idx in range(self.eps_iso.shape[0]):
            layer = self.materialize_layer(layer_idx)
            base[layer_idx, 0] = layer[0, 0, 0, 0]
            base[layer_idx, 1] = layer[0, 0, 1, 1]
        return base

    def __len__(self) -> int:
        """Return number of propagation slices."""
        return self.eps_iso.shape[0]

    def _constant_direction_map(self) -> NDArray[np.float64]:
        """Return a constant direction map for one layer."""
        ny, nx = self.eps_iso.shape[1:3]
        return np.broadcast_to(self.beam_direction, (ny, nx, 3))

    @staticmethod
    def _transverse_basis(
        k_map: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Return local transverse basis vectors for a ``(..., 3)`` direction map."""
        k_map = np.asarray(k_map, dtype=float)
        norm = np.linalg.norm(k_map, axis=-1, keepdims=True)
        k = np.divide(k_map, norm, out=np.zeros_like(k_map), where=norm > 0)
        lab_x = np.array([1.0, 0.0, 0.0])
        lab_y = np.array([0.0, 1.0, 0.0])
        use_y = np.abs(np.einsum("...i,i->...", k, lab_x)) > 0.95
        reference = np.broadcast_to(lab_x, k.shape).copy()
        reference[use_y] = lab_y
        e1 = reference - np.einsum("...i,...i->...", reference, k)[..., None] * k
        e1_norm = np.linalg.norm(e1, axis=-1, keepdims=True)
        e1 = np.divide(e1, e1_norm, out=np.zeros_like(e1), where=e1_norm > 0)
        e2 = np.cross(k, e1)
        e2_norm = np.linalg.norm(e2, axis=-1, keepdims=True)
        e2 = np.divide(e2, e2_norm, out=np.zeros_like(e2), where=e2_norm > 0)
        return e1, e2, k

    def project_layer(
        self,
        layer_idx: int,
        k_map: NDArray[np.float64] | None = None,
    ) -> NDArray[np.complex128]:
        """Project one layer onto the local transverse Jones basis."""
        if k_map is None:
            k_map = self._constant_direction_map()
        k_map = np.asarray(k_map, dtype=float)
        if k_map.shape != (*self.eps_iso.shape[1:3], 3):
            raise ValueError(
                "k_map must have shape (Ny, Nx, 3), got "
                f"{k_map.shape} for layer shape {self.eps_iso.shape[1:3]}."
            )
        e1, e2, k = self._transverse_basis(k_map)
        eps_iso = self.eps_iso[layer_idx]
        g = self.gyrotropic_vector[layer_idx]
        q = self.linear_tensor[layer_idx]

        gk = np.einsum("...i,...i->...", g, k)
        q11 = np.einsum("...i,...ij,...j->...", e1, q, e1)
        q22 = np.einsum("...i,...ij,...j->...", e2, q, e2)
        q12 = np.einsum("...i,...ij,...j->...", e1, q, e2)

        eps = np.zeros((*eps_iso.shape, 2, 2), dtype=self.eps_iso.dtype)
        eps[..., 0, 0] = eps_iso + q11 - q22
        eps[..., 1, 1] = eps_iso - q11 + q22
        eps[..., 0, 1] = 1.0j * gk + 2.0 * q12
        eps[..., 1, 0] = -1.0j * gk + 2.0 * q12
        return eps

    def materialize_layer(self, layer_idx: int) -> NDArray[np.complex128]:
        """Return one layer projected along the nominal beam direction."""
        return self.project_layer(layer_idx)

    def materialize(self) -> NDArray[np.complex128]:
        """Return a dense tensor stack projected along the nominal direction."""
        return np.stack([self.materialize_layer(iz) for iz in range(len(self))])


@dataclass(frozen=True)
class Layer:
    """A single material layer in a multilayer stack.

    Attributes
    ----------
    material : str
        Name of the material (e.g. ``"Pt"``, ``"Co"``).
    thickness : float
        Physical thickness in metres.
    """

    material: str
    thickness: float
    components: Tuple[Tuple[str, float], ...] | None = None

    def __post_init__(self) -> None:
        """Handle the internal post init operation.

        Parameters
        ----------
        None
            This function takes no explicit input parameters.

        Returns
        -------
        None
            The function completes in place.
        """
        if self.components is None:
            object.__setattr__(
                self,
                "components",
                ((self.material, self.thickness),),
            )

    @property
    def is_composite(self) -> bool:
        """Return ``True`` when this layer combines multiple materials.

        Parameters
        ----------
        None
            This function takes no explicit input parameters.

        Returns
        -------
        result : bool
            Return value produced by the function.
        """
        return len(self.components or ()) > 1

    # @property
    # def thickness(self) -> float:
    #    return self.thickness_nm * 1e-9

    def to_txt_line(self) -> str:
        """Run the to txt line operation.

        Parameters
        ----------
        None
            This function takes no explicit input parameters.

        Returns
        -------
        result : str
            Return value produced by the function.
        """
        return f"{self.material} {format_nm_as_meter_string(self.thickness_)}"


@dataclass
class MultilayerRecipe:
    """Parsed representation of a multilayer recipe string.

    Attributes
    ----------
    recipe_string : str
        Original recipe string as supplied by the user.
    layers : list of Layer
        Fully expanded list of layers in deposition order.
    sample_name : str or None
        Optional human-readable sample identifier.
    comments : list of str
        Optional free-text annotations attached to the recipe.
    """

    recipe_string: str
    layers: List[Layer]
    sample_name: str | None = None
    comments: List[str] = field(default_factory=list)

    # @property
    # def total_thickness_nm(self) -> float:
    #    return sum(layer.thickness_nm for layer in self.layers)

    @property
    def total_thickness(self) -> float:
        """Total physical thickness of the stack in metres.

        Parameters
        ----------
        None
            This function takes no explicit input parameters.

        Returns
        -------
        result : float
            Return value produced by the function.
        """
        return sum(layer.thickness for layer in self.layers)

    def add_comment(self, text: str) -> None:
        """Append a free-text comment to the recipe.

        Parameters
        ----------
        text : str
            Comment string to append.

        Returns
        -------
        None
            The function completes in place.
        """
        self.comments.append(text)

    def summary(self) -> str:
        """Return a human-readable summary of the recipe.

        Returns
        -------
        str
            Multi-line string listing sample name, recipe, layer count,
            total thickness, and any attached comments.

        Parameters
        ----------
        None
            This function takes no explicit input parameters.
        """
        lines = []
        if self.sample_name:
            lines.append(f"Sample: {self.sample_name}")
        lines.append(f"Recipe: {self.recipe_string}")
        lines.append(f"Number of layers: {len(self.layers)}")
        # lines.append(f"Total thickness: {self.total_thickness_nm:.6g} nm")
        lines.append(f"Total thickness: {self.total_thickness:.6g} m")
        if self.comments:
            lines.append("Comments:")
            for comment in self.comments:
                lines.append(f"  - {comment}")
        return "\n".join(lines)


# ============================================================
# Formatting helpers
# ============================================================


def format_nm_as_meter_string(value_nm: float) -> str:
    """Convert thickness in nm to a compact string in meters.
    Examples:
        5    -> '5e-9'
        1.5  -> '1.5e-9'
        0.25 -> '0.25e-9'

    Parameters
    ----------
    value_nm : float
        Input value for ``value_nm``.

    Returns
    -------
    result : str
        Return value produced by the function.
    """
    if float(value_nm).is_integer():
        return f"{int(value_nm)}e-9"
    return f"{value_nm:g}e-9"


# ============================================================
# Parser
# ============================================================


class RecipeParser:
    """
    Parser for multilayer recipes.

    Supported syntax:
        Pt(5)
        Pt(5)/Co(1)
        Pt(4)Co(6)
        Pt(5)/[Pt(2)/Co(1)]x10/Ta(5)
        [Pt(2)/[Co(1)/Ni(0.5)]x3]x5

    Conventions:
    - Thickness is assumed to be in nm
    - Layers are separated by '/'
    - Adjacent material terms without '/' form one effective-medium layer
    - Repeated blocks use [ ... ]xN
    - Nested repeated blocks are supported
    """

    def __init__(self, text: str):
        """Initialize a RecipeParser instance.

        Parameters
        ----------
        text : str
            Input value for ``text``.

        Returns
        -------
        None
            The function completes in place.
        """
        self.text = text.replace(" ", "")
        self.pos = 0

    def parse(self) -> List[Layer]:
        """Parse the recipe string and return the expanded list of layers.

        Returns
        -------
        list of Layer
            Fully expanded layer sequence in deposition order.

        Raises
        ------
        ValueError
            If the recipe string contains invalid syntax.

        Parameters
        ----------
        None
            This function takes no explicit input parameters.
        """
        layers = self._parse_sequence(stop_char=None)
        if self.pos != len(self.text):
            raise ValueError(
                f"Unexpected trailing content at position {self.pos}: "
                f"{self.text[self.pos:]}"
            )
        return layers

    def _parse_sequence(self, stop_char: str | None) -> List[Layer]:
        """Handle the internal parse sequence operation.

        Parameters
        ----------
        stop_char : str | None
            Input value for ``stop_char``.

        Returns
        -------
        result : List[Layer]
            Return value produced by the function.
        """
        layers: List[Layer] = []

        while self.pos < len(self.text):
            char = self.text[self.pos]

            if stop_char is not None and char == stop_char:
                break

            if char == "/":
                self.pos += 1
                continue

            if char == "[":
                block_layers = self._parse_block()
                layers.extend(block_layers)
                continue

            layer = self._parse_composite_layer()
            layers.append(layer)

        return layers

    def _parse_block(self) -> List[Layer]:
        """Handle the internal parse block operation.

        Parameters
        ----------
        None
            This function takes no explicit input parameters.

        Returns
        -------
        result : List[Layer]
            Return value produced by the function.
        """
        self._expect("[")
        inner_layers = self._parse_sequence(stop_char="]")
        self._expect("]")

        self._expect("x")
        repeat = self._parse_integer()

        return inner_layers * repeat

    def _parse_composite_layer(self) -> Layer:
        """Handle the internal parse composite layer operation.

        Parameters
        ----------
        None
            This function takes no explicit input parameters.

        Returns
        -------
        result : Layer
            Return value produced by the function.
        """
        components: list[Layer] = [self._parse_layer()]
        while self.pos < len(self.text):
            char = self.text[self.pos]
            if char in "/]":
                break
            if char == "[":
                break
            components.append(self._parse_layer())

        if len(components) == 1:
            return components[0]

        total_thickness = sum(component.thickness for component in components)
        label = "".join(
            f"{component.material}({component.thickness * 1e9:g})"
            for component in components
        )
        return Layer(
            material=label,
            thickness=total_thickness,
            components=tuple(
                (component.material, component.thickness)
                for component in components
            ),
        )

    def _parse_layer(self) -> Layer:
        """Handle the internal parse layer operation.

        Parameters
        ----------
        None
            This function takes no explicit input parameters.

        Returns
        -------
        result : Layer
            Return value produced by the function.
        """
        material = self._parse_material()
        self._expect("(")
        thickness_nm = self._parse_number()
        self._expect(")")

        if thickness_nm <= 0:
            raise ValueError(
                f"Thickness must be positive for material '{material}', "
                f"got {thickness_nm}"
            )

        return Layer(material=material, thickness=thickness_nm * 1e-9)

    def _parse_material(self) -> str:
        """Handle the internal parse material operation.

        Parameters
        ----------
        None
            This function takes no explicit input parameters.

        Returns
        -------
        result : str
            Return value produced by the function.
        """
        start = self.pos
        while self.pos < len(self.text):
            char = self.text[self.pos]
            if char.isalnum() or char == "_":
                self.pos += 1
            else:
                break

        if self.pos == start:
            raise ValueError(f"Expected material at position {self.pos}")

        return self.text[start : self.pos]

    def _parse_number(self) -> float:
        """Handle the internal parse number operation.

        Parameters
        ----------
        None
            This function takes no explicit input parameters.

        Returns
        -------
        result : float
            Return value produced by the function.
        """
        start = self.pos
        dot_count = 0

        while self.pos < len(self.text):
            char = self.text[self.pos]
            if char.isdigit():
                self.pos += 1
            elif char == ".":
                dot_count += 1
                if dot_count > 1:
                    raise ValueError(f"Invalid number at position {start}")
                self.pos += 1
            else:
                break

        if self.pos == start:
            raise ValueError(f"Expected number at position {self.pos}")

        value_str = self.text[start : self.pos]
        try:
            return float(value_str)
        except ValueError as exc:
            raise ValueError(f"Invalid number '{value_str}'") from exc

    def _parse_integer(self) -> int:
        """Handle the internal parse integer operation.

        Parameters
        ----------
        None
            This function takes no explicit input parameters.

        Returns
        -------
        result : int
            Return value produced by the function.
        """
        start = self.pos
        while self.pos < len(self.text) and self.text[self.pos].isdigit():
            self.pos += 1

        if self.pos == start:
            raise ValueError(f"Expected integer at position {self.pos}")

        value = int(self.text[start : self.pos])
        if value <= 0:
            raise ValueError(f"Repeat count must be positive, got {value}")
        return value

    def _expect(self, token: str) -> None:
        """Handle the internal expect operation.

        Parameters
        ----------
        token : str
            Input value for ``token``.

        Returns
        -------
        None
            The function completes in place.
        """
        if self.pos >= len(self.text) or self.text[self.pos] != token:
            found = self.text[self.pos] if self.pos < len(self.text) else "EOF"
            raise ValueError(
                f"Expected '{token}' at position {self.pos}, found '{found}'"
            )
        self.pos += 1


# ============================================================
# Public API
# ============================================================


def parse_recipe(
    recipe: str,
    sample_name: str | None = None,
    comments: List[str] | None = None,
) -> MultilayerRecipe:
    """Parse a multilayer recipe string into a :class:`MultilayerRecipe`.

    Parameters
    ----------
    recipe : str
        Recipe string, e.g. ``"Pt(5)/[Co(2)/Pt(1)]x10/Ta(3)"``.
        Thicknesses are in nanometres; brackets with ``xN`` denote repetitions.
        Adjacent material terms without a slash, e.g. ``"Pt(4)Co(6)"``, are
        parsed as one effective-medium layer with thickness-weighted material
        properties.
    sample_name : str or None, optional
        Human-readable label attached to the returned recipe.
    comments : list of str or None, optional
        Free-text annotations attached to the returned recipe.

    Returns
    -------
    MultilayerRecipe
        Dataclass with the fully expanded layer list.

    Raises
    ------
    ValueError
        If the recipe string contains invalid syntax.
    """
    parser = RecipeParser(recipe)
    layers = parser.parse()

    return MultilayerRecipe(
        recipe_string=recipe,
        layers=layers,
        sample_name=sample_name,
        comments=comments[:] if comments else [],
    )


class Structure:
    """Layered thin-film structure for coherent scattering simulations.

    Layers are added sequentially via :meth:`add_layer`. The structure stores
    per-layer thicknesses, refractive indices, and effective indices
    (``n * thickness``) for downstream scattering calculations.

    Parameters
    ----------
    name : str
        Human-readable label for the structure.
    material_params : material_params
        Material database providing refractive indices for each element.

    Attributes
    ----------
    name : str
        Structure label.
    material_params : material_params
        Reference to the material database.
    sample_shape :
        Shape of the sample in pixels.
    real_space_pixel_size : float
        Physical pixel size (lateral, xy plane) in metres.
    layer_names : list of str
        Element name of each layer in deposition order.
    layer_thicknesses : list of float
        Physical thickness of each layer in metres.
    layer_refractive_indices : list of complex
        Complex refractive index of each layer.
    effective_refractive_indices : list of complex
        Effective index (``n * thickness``) of each layer.
    dielectric_tensors : (Nz,Ny,Nx,2,2) complex ndarray
    magnetization : (Nz,Ny,Nx,3) float ndarray
    mask : (Nz,Ny,Nx) float ndarray for mask. if 1 the material is there if 0 vacuum. Used for aperture function or magnetic tracks
    """

    def __init__(
        self,
        name: str,
        material_params: material_params,
        sample_shape: list[int, int, int],
        real_space_pixel_size: float,
    ) -> None:
        """Initialize a Structure instance.

        Parameters
        ----------
        name : str
            Input value for ``name``.
        material_params : material_params
            Input value for ``material_params``.
        sample_shape : list[int, int, int]
            Input value for ``sample_shape``.
        real_space_pixel_size : float
            Input value for ``real_space_pixel_size``.

        Returns
        -------
        None
            The function completes in place.
        """
        self.name = name
        self.material_params = material_params
        self.sample_shape = sample_shape
        self.real_space_pixel_size = real_space_pixel_size
        self.layer_names: list[str] = []
        self.layer_thicknesses: list[float] = []
        self.layer_refractive_indices: list[complex] = []
        self.effective_refractive_indices: list[complex] = []
        self.dielectric_tensors: list[NDArray[np.complex128]] = (
            []
        )  # Optional: store dielectric tensors if needed
        self.effective_dielectric_tensors: list[NDArray[np.complex128]] = (
            []
        )  # Optional: store effective dielectric tensors if needed
        self.magnetization: NDArray[np.float64] | None = (
            None  # Optional: store magnetization map
        )
        self.mask: NDArray[np.float64] | None = (
            None  # Optional: store sample mask for aperture function
        )
        self.sample_tilt_theta: float = 0.0
        self.sample_tilt_axis: Literal["x", "y"] = "x"
        self.sample_tilt_voxel_size: float | None = None
        self.sample_tilt_antialias_samples: int = 3
        self.sample_tilt_simulation_z_extent: float | None = None
        self.sample_tilt_center_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)

        if self.sample_shape[0] == 0:
            self.sample_shape[0] = len(self.layer_names)

        self.calc_real_space_coordinates()
        self.get_extent_real_space()

    @property
    def propagation_layer_thicknesses(self) -> list[float]:
        """Return layer thicknesses for the current propagation stack."""
        return getattr(self, "_propagation_layer_thicknesses", self.layer_thicknesses)

    def set_sample_tilt(
        self,
        theta: float = 0.0,
        axis: Literal["x", "y"] = "x",
        voxel_size: float | None = None,
        antialias_samples: int = 3,
        simulation_z_extent: float | None = None,
        center_offset: ArrayLike | None = None,
    ) -> None:
        """Configure a tilted material stack relative to simulation z slices.

        ``theta`` is in radians. ``theta=0`` preserves the original one-slice-
        per-layer representation. Non-zero values voxelize the multilayer into
        z planes parallel to light propagation and add vacuum before/after the
        tilted film as needed. ``center_offset`` shifts the film centre in lab
        ``(y, x, z)`` coordinates relative to the simulation-volume centre.
        """
        if axis not in ("x", "y"):
            raise ValueError(f"tilt axis must be 'x' or 'y', got {axis!r}")
        if voxel_size is not None and voxel_size <= 0:
            raise ValueError(f"tilt voxel_size must be positive, got {voxel_size}")
        if simulation_z_extent is not None and simulation_z_extent <= 0:
            raise ValueError(
                f"simulation_z_extent must be positive, got {simulation_z_extent}"
            )
        if center_offset is None:
            center_offset_tuple = (0.0, 0.0, 0.0)
        else:
            center_offset_arr = np.asarray(center_offset, dtype=float)
            if center_offset_arr.shape != (3,):
                raise ValueError(
                    "center_offset must contain three values in lab (y, x, z) "
                    f"coordinates, got shape {center_offset_arr.shape}."
                )
            if not np.all(np.isfinite(center_offset_arr)):
                raise ValueError(f"center_offset must be finite, got {center_offset!r}.")
            center_offset_tuple = tuple(float(v) for v in center_offset_arr)
        self.sample_tilt_theta = float(theta)
        self.sample_tilt_axis = axis
        self.sample_tilt_voxel_size = voxel_size
        self.sample_tilt_antialias_samples = max(1, int(antialias_samples))
        self.sample_tilt_simulation_z_extent = simulation_z_extent
        self.sample_tilt_center_offset = center_offset_tuple
        if hasattr(self, "_propagation_layer_thicknesses"):
            delattr(self, "_propagation_layer_thicknesses")

    def _tilted_layer_fractions(self) -> tuple[NDArray[np.float64], list[float]]:
        """Return fractional layer occupancy for tilted sample voxels."""
        theta = float(getattr(self, "sample_tilt_theta", 0.0))
        center_offset = getattr(self, "sample_tilt_center_offset", (0.0, 0.0, 0.0))
        offset_y, offset_x, offset_z = (float(v) for v in center_offset)
        has_center_offset = not np.allclose((offset_y, offset_x, offset_z), 0.0)
        if (
            np.isclose(theta, 0.0)
            and self.sample_tilt_simulation_z_extent is None
            and not has_center_offset
        ):
            nz = len(self.layer_thicknesses)
            fractions = np.zeros((*self.mask.shape, nz), dtype=float)
            for layer_idx in range(nz):
                fractions[layer_idx, ..., layer_idx] = 1.0
            return fractions, list(self.layer_thicknesses)

        cos_theta = float(np.cos(theta))
        sin_theta = float(np.sin(theta))
        layer_edges = np.concatenate(([0.0], np.cumsum(self.layer_thicknesses)))
        total_thickness = float(layer_edges[-1])
        dz = self.sample_tilt_voxel_size or min(self.layer_thicknesses)

        ny, nx = self.sample_shape[1], self.sample_shape[2]
        if self.sample_tilt_axis == "x":
            lateral = (
                np.arange(nx, dtype=float) - nx / 2 + 0.5
            ) * self.real_space_pixel_size - offset_x
            lateral_shape = (1, 1, nx)
        else:
            lateral = (
                np.arange(ny, dtype=float) - ny / 2 + 0.5
            ) * self.real_space_pixel_size - offset_y
            lateral_shape = (1, ny, 1)
        if self.sample_tilt_simulation_z_extent is None:
            if abs(cos_theta) < 1e-9:
                raise ValueError(
                    "simulation_z_extent is required for sample_tilt_theta near 90 degrees."
                )
            shifts = lateral * sin_theta
            z_min = (0.0 - np.max(shifts)) / cos_theta
            z_max = (total_thickness - np.min(shifts)) / cos_theta
            nz = int(np.ceil((z_max - z_min) / dz))
            z_edges = z_min + np.arange(nz + 1, dtype=float) * dz
            material_depth_offset = 0.0
        else:
            nz = int(np.ceil(float(self.sample_tilt_simulation_z_extent) / dz))
            z_edges = (np.arange(nz + 1, dtype=float) - nz / 2.0) * dz
            material_depth_offset = 0.5 * total_thickness
        z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])
        thicknesses = np.diff(z_edges).astype(float).tolist()

        fractions = np.zeros((nz, ny, nx, len(self.layer_thicknesses)), dtype=float)
        n_samples = max(1, int(self.sample_tilt_antialias_samples))
        offsets = (np.arange(n_samples, dtype=float) + 0.5) / n_samples - 0.5
        sample_weight = 1.0 / (n_samples * n_samples)
        lateral_grid = lateral.reshape(lateral_shape)
        for z_offset in offsets * dz:
            z_sample = (z_centers + z_offset - offset_z)[:, None, None]
            for lateral_offset in offsets * self.real_space_pixel_size:
                material_depth = (
                    material_depth_offset
                    + z_sample * cos_theta
                    + (lateral_grid + lateral_offset) * sin_theta
                )
                for layer_idx, (z0, z1) in enumerate(zip(layer_edges[:-1], layer_edges[1:])):
                    inside = (material_depth >= z0) & (material_depth < z1)
                    fractions[..., layer_idx] += inside * sample_weight
        return fractions, thicknesses

    def tilted_material_coordinate_grids(
        self,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Return coordinates attached to the tilted material stack.

        Returns
        -------
        film_y, film_x, material_depth : ndarray
            Arrays of shape ``(Nz, Ny, Nx)``. ``film_x`` and ``film_y`` are
            coordinates parallel to the film planes; ``material_depth`` is the
            coordinate normal to the film, increasing through the layer stack.
        """
        theta = float(getattr(self, "sample_tilt_theta", 0.0))
        cos_theta = float(np.cos(theta))
        sin_theta = float(np.sin(theta))
        total_thickness = float(np.sum(self.layer_thicknesses))
        dz = self.sample_tilt_voxel_size or min(self.layer_thicknesses)
        ny, nx = self.sample_shape[1], self.sample_shape[2]
        y = (np.arange(ny, dtype=float) - ny / 2 + 0.5) * self.real_space_pixel_size
        x = (np.arange(nx, dtype=float) - nx / 2 + 0.5) * self.real_space_pixel_size
        center_offset = getattr(self, "sample_tilt_center_offset", (0.0, 0.0, 0.0))
        offset_y, offset_x, offset_z = (float(v) for v in center_offset)
        has_center_offset = not np.allclose((offset_y, offset_x, offset_z), 0.0)

        if (
            np.isclose(theta, 0.0)
            and self.sample_tilt_simulation_z_extent is None
            and not has_center_offset
        ):
            z_centers = np.cumsum(self.layer_thicknesses) - 0.5 * np.asarray(
                self.layer_thicknesses
            )
            film_y = np.broadcast_to(y[None, :, None] - offset_y, (len(z_centers), ny, nx))
            film_x = np.broadcast_to(x[None, None, :] - offset_x, (len(z_centers), ny, nx))
            material_depth = np.broadcast_to(
                z_centers[:, None, None] - offset_z, (len(z_centers), ny, nx)
            )
            return film_y, film_x, material_depth

        if self.sample_tilt_axis == "x":
            lateral = (np.arange(nx, dtype=float) - nx / 2 + 0.5) * self.real_space_pixel_size - offset_x
            shifts = lateral * sin_theta
        else:
            lateral = (np.arange(ny, dtype=float) - ny / 2 + 0.5) * self.real_space_pixel_size - offset_y
            shifts = lateral * sin_theta

        if self.sample_tilt_simulation_z_extent is None:
            if abs(cos_theta) < 1e-9:
                raise ValueError(
                    "simulation_z_extent is required for sample_tilt_theta near 90 degrees."
                )
            z_min = (0.0 - np.max(shifts)) / cos_theta
            z_max = (total_thickness - np.min(shifts)) / cos_theta
            nz = int(np.ceil((z_max - z_min) / dz))
            z_centers = z_min + (np.arange(nz, dtype=float) + 0.5) * dz
            material_depth_offset = 0.0
        else:
            nz = int(np.ceil(float(self.sample_tilt_simulation_z_extent) / dz))
            z_centers = (np.arange(nz, dtype=float) - nz / 2 + 0.5) * dz
            material_depth_offset = 0.5 * total_thickness

        z_grid = z_centers[:, None, None] - offset_z
        y_grid = y[None, :, None] - offset_y
        x_grid = x[None, None, :] - offset_x
        if self.sample_tilt_axis == "x":
            film_x = x_grid * cos_theta - z_grid * sin_theta
            film_y = np.broadcast_to(y_grid, (nz, ny, nx))
            material_depth = material_depth_offset + z_grid * cos_theta + x_grid * sin_theta
            film_x = np.broadcast_to(film_x, (nz, ny, nx))
            material_depth = np.broadcast_to(material_depth, (nz, ny, nx))
        else:
            film_y = y_grid * cos_theta - z_grid * sin_theta
            film_x = np.broadcast_to(x_grid, (nz, ny, nx))
            material_depth = material_depth_offset + z_grid * cos_theta + y_grid * sin_theta
            film_y = np.broadcast_to(film_y, (nz, ny, nx))
            material_depth = np.broadcast_to(material_depth, (nz, ny, nx))
        return film_y, film_x, material_depth

    @staticmethod
    def _tilted_layer_magnetization(
        tilted_magnetization: NDArray[np.float64] | None,
        fallback_magnetization: NDArray[np.float64],
        layer_idx: int,
        tilted_shape: tuple[int, int, int],
    ) -> NDArray[np.float64]:
        """Return magnetization on the tilted voxel grid for one layer."""
        if tilted_magnetization is None:
            return np.broadcast_to(
                fallback_magnetization[layer_idx][None, ...],
                (*tilted_shape, 3),
            )
        tilted_magnetization = np.asarray(tilted_magnetization, dtype=float)
        if tilted_magnetization.shape == (*tilted_shape, 3):
            return tilted_magnetization
        if tilted_magnetization.shape[0] > layer_idx and tilted_magnetization.shape[1:] == (
            *tilted_shape,
            3,
        ):
            return tilted_magnetization[layer_idx]
        raise ValueError(
            "tilted_magnetization must have shape (Nz, Ny, Nx, 3) or "
            "(Nlayer, Nz, Ny, Nx, 3)."
        )

    def _tilted_dense_dielectric_tensor(self) -> NDArray[np.complex128]:
        """Voxelize the material stack into tilted, interpolated dielectric slices."""
        if self.mask is None or self.magnetization is None:
            raise ValueError("mask and magnetization must be assigned before tilt voxelization.")

        fractions, thicknesses = self._tilted_layer_fractions()
        mask = np.asarray(self.mask, dtype=float)
        magnetization = np.asarray(self.magnetization, dtype=float)
        tilted_magnetization = getattr(self, "tilted_magnetization", None)
        dt = np.asarray(self.dielectric_tensors)
        nz, ny, nx, n_layers = fractions.shape
        out = np.zeros((nz, ny, nx, 2, 2), dtype=dt.dtype)
        material_fraction = np.zeros((nz, ny, nx), dtype=float)
        tol = 1e-14

        for layer_idx in range(n_layers):
            layer_fraction = fractions[..., layer_idx] * mask[layer_idx]
            material_fraction += layer_fraction
            if not np.any(layer_fraction > tol):
                continue
            eps = np.zeros((ny, nx, 2, 2), dtype=dt.dtype)
            eps[..., 0, 0] = dt[layer_idx, 0, 0, 0]
            eps[..., 1, 1] = dt[layer_idx, 0, 1, 1]
            layer_m = self._tilted_layer_magnetization(
                tilted_magnetization,
                magnetization,
                layer_idx,
                (nz, ny, nx),
            )
            if np.any(np.abs(dt[layer_idx, 1]) > tol):
                eps = eps[None, ...] + layer_m[..., 2, None, None] * dt[layer_idx, 1]
            else:
                eps = eps[None, ...]
            if np.any(np.abs(dt[layer_idx, 2]) > tol):
                mx = layer_m[..., 0]
                my = layer_m[..., 1]
                eps += (np.abs(mx) ** 2 - np.abs(my) ** 2)[..., None, None] * dt[layer_idx, 2]
            out += layer_fraction[..., None, None] * eps

        vacuum_fraction = np.clip(1.0 - material_fraction, 0.0, 1.0)
        out[..., 0, 0] += vacuum_fraction
        out[..., 1, 1] += vacuum_fraction
        self._propagation_layer_thicknesses = thicknesses
        self.aperture_support_regions = None
        return out

    def _tilted_dense_scalar_refractive_index(self, pol) -> NDArray[np.complex128]:
        """Voxelize the material stack into tilted scalar refractive-index slices."""
        from scattering_calculator.beam_propagator.simple_propagation import (
            scalar_refractive_index_coefficients,
        )

        if self.mask is None or self.magnetization is None:
            raise ValueError("mask and magnetization must be assigned before tilt voxelization.")

        fractions, thicknesses = self._tilted_layer_fractions()
        mask = np.asarray(self.mask, dtype=float)
        magnetization = np.asarray(self.magnetization, dtype=float)
        tilted_magnetization = getattr(self, "tilted_magnetization", None)
        indices = np.asarray(self.layer_refractive_indices, dtype=complex)
        circular_coeff, linear_coeff = scalar_refractive_index_coefficients(pol)
        out = np.zeros(fractions.shape[:3], dtype=complex)
        material_fraction = np.zeros(fractions.shape[:3], dtype=float)
        tol = 1e-14

        for layer_idx in range(fractions.shape[-1]):
            layer_fraction = fractions[..., layer_idx] * mask[layer_idx]
            material_fraction += layer_fraction
            if not np.any(layer_fraction > tol):
                continue
            layer_m = self._tilted_layer_magnetization(
                tilted_magnetization,
                magnetization,
                layer_idx,
                fractions.shape[:3],
            )
            n_layer = np.full(fractions.shape[:3], indices[layer_idx, 0], dtype=complex)
            if abs(circular_coeff * indices[layer_idx, 1]) > tol:
                n_layer += (
                    circular_coeff
                    * layer_m[..., 2]
                    * indices[layer_idx, 1]
                )
            if abs(linear_coeff * indices[layer_idx, 2]) > tol:
                mx = layer_m[..., 0]
                my = layer_m[..., 1]
                n_layer += (
                    linear_coeff
                    * (np.abs(mx) ** 2 - np.abs(my) ** 2)
                    * indices[layer_idx, 2]
                )
            out += layer_fraction * n_layer

        out += np.clip(1.0 - material_fraction, 0.0, 1.0)
        self._propagation_layer_thicknesses = thicknesses
        self.aperture_support_regions = None
        return out

    @staticmethod
    def _normalize_beam_direction(beam_direction: ArrayLike) -> NDArray[np.float64]:
        """Return a unit beam direction in lab ``(x, y, z)`` coordinates."""
        direction = np.asarray(beam_direction, dtype=float)
        if direction.shape != (3,):
            raise ValueError(
                "beam_direction must be a three-value vector in lab (x, y, z) "
                f"coordinates, got shape {direction.shape}."
            )
        norm = np.linalg.norm(direction)
        if not np.isfinite(norm) or norm == 0.0:
            raise ValueError(f"beam_direction must be finite and nonzero, got {direction!r}.")
        return direction / norm

    @classmethod
    def _beam_transverse_basis(
        cls,
        beam_direction: ArrayLike,
        beam_polarization_basis: tuple[ArrayLike, ArrayLike] | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Return orthonormal Jones basis vectors ``(e1, e2, k)``.

        The basis uses lab ``(x, y, z)`` coordinates. ``e1`` and ``e2`` are
        transverse to the normalized beam direction ``k`` and are chosen so that
        ``e1 x e2 = k``. With ``k = z``, the default basis is ``e1 = x`` and
        ``e2 = y``.
        """
        k = cls._normalize_beam_direction(beam_direction)
        if beam_polarization_basis is not None:
            e1 = np.asarray(beam_polarization_basis[0], dtype=float)
            e2 = np.asarray(beam_polarization_basis[1], dtype=float)
            if e1.shape != (3,) or e2.shape != (3,):
                raise ValueError("beam_polarization_basis vectors must each have shape (3,).")
            e1 = e1 - np.dot(e1, k) * k
            e1_norm = np.linalg.norm(e1)
            if not np.isfinite(e1_norm) or e1_norm == 0.0:
                raise ValueError("First beam_polarization_basis vector is parallel to beam.")
            e1 = e1 / e1_norm
            e2 = e2 - np.dot(e2, k) * k - np.dot(e2, e1) * e1
            e2_norm = np.linalg.norm(e2)
            if not np.isfinite(e2_norm) or e2_norm == 0.0:
                raise ValueError(
                    "Second beam_polarization_basis vector is not an independent "
                    "transverse direction."
                )
            e2 = e2 / e2_norm
            if np.dot(np.cross(e1, e2), k) < 0:
                e2 = -e2
            return e1, e2, k

        lab_x = np.array([1.0, 0.0, 0.0])
        lab_y = np.array([0.0, 1.0, 0.0])
        reference = lab_x if abs(np.dot(k, lab_x)) < 0.95 else lab_y
        e1 = reference - np.dot(reference, k) * k
        e1 = e1 / np.linalg.norm(e1)
        e2 = np.cross(k, e1)
        e2 = e2 / np.linalg.norm(e2)
        return e1, e2, k

    def _dense_dielectric_tensor_for_beam_direction(
        self,
        beam_direction: ArrayLike,
        beam_polarization_basis: tuple[ArrayLike, ArrayLike] | None = None,
    ) -> NDArray[np.complex128]:
        """Build a dense Jones tensor projected perpendicular to the beam."""
        if self.mask is None or self.magnetization is None:
            raise ValueError("mask and magnetization must be assigned before tensor projection.")

        e1, e2, k = self._beam_transverse_basis(
            beam_direction,
            beam_polarization_basis=beam_polarization_basis,
        )
        fractions, thicknesses = self._tilted_layer_fractions()
        mask = np.asarray(self.mask, dtype=float)
        magnetization = np.asarray(self.magnetization, dtype=float)
        tilted_magnetization = getattr(self, "tilted_magnetization", None)
        indices = np.asarray(self.layer_refractive_indices, dtype=complex)
        nz, ny, nx, n_layers = fractions.shape
        out = np.zeros((nz, ny, nx, 2, 2), dtype=complex)
        material_fraction = np.zeros((nz, ny, nx), dtype=float)
        tol = 1e-14

        for layer_idx in range(n_layers):
            layer_fraction = fractions[..., layer_idx] * mask[layer_idx]
            material_fraction += layer_fraction
            if not np.any(layer_fraction > tol):
                continue

            n0, dn_c, dn_l = indices[layer_idx]
            eps0 = n0**2
            eps_c = 2.0 * n0 * dn_c
            eps_l = 2.0 * n0 * dn_l
            layer_m = self._tilted_layer_magnetization(
                tilted_magnetization,
                magnetization,
                layer_idx,
                (nz, ny, nx),
            )
            m1 = np.einsum("...i,i->...", layer_m, e1)
            m2 = np.einsum("...i,i->...", layer_m, e2)
            mk = np.einsum("...i,i->...", layer_m, k)

            eps = np.zeros((nz, ny, nx, 2, 2), dtype=complex)
            eps[..., 0, 0] = eps0
            eps[..., 1, 1] = eps0
            if abs(eps_c) > tol:
                eps[..., 0, 1] += 1.0j * eps_c * mk
                eps[..., 1, 0] += -1.0j * eps_c * mk
            if abs(eps_l) > tol:
                linear_diag = eps_l * (np.abs(m1) ** 2 - np.abs(m2) ** 2)
                linear_offdiag = eps_l * 2.0 * m1 * m2
                eps[..., 0, 0] += linear_diag
                eps[..., 1, 1] -= linear_diag
                eps[..., 0, 1] += linear_offdiag
                eps[..., 1, 0] += linear_offdiag
            out += layer_fraction[..., None, None] * eps

        vacuum_fraction = np.clip(1.0 - material_fraction, 0.0, 1.0)
        out[..., 0, 0] += vacuum_fraction
        out[..., 1, 1] += vacuum_fraction
        self._propagation_layer_thicknesses = thicknesses
        self.aperture_support_regions = None
        self.beam_direction = k
        self.beam_polarization_basis = (e1, e2)
        return out

    def _dynamic_projected_dielectric_tensor_stack(
        self,
        beam_direction: ArrayLike | None = None,
    ) -> DynamicProjectedDielectricTensorStack:
        """Build direction-independent channels for local-k Jones projection."""
        if self.mask is None or self.magnetization is None:
            raise ValueError("mask and magnetization must be assigned before tensor projection.")

        if beam_direction is None:
            beam_direction = (0.0, 0.0, 1.0)
        k = self._normalize_beam_direction(beam_direction)
        fractions, thicknesses = self._tilted_layer_fractions()
        mask = np.asarray(self.mask, dtype=float)
        magnetization = np.asarray(self.magnetization, dtype=float)
        tilted_magnetization = getattr(self, "tilted_magnetization", None)
        indices = np.asarray(self.layer_refractive_indices, dtype=complex)
        nz, ny, nx, n_layers = fractions.shape
        eps_iso = np.zeros((nz, ny, nx), dtype=complex)
        gyrotropic_vector = np.zeros((nz, ny, nx, 3), dtype=complex)
        linear_tensor = np.zeros((nz, ny, nx, 3, 3), dtype=complex)
        material_fraction = np.zeros((nz, ny, nx), dtype=float)
        tol = 1e-14

        for layer_idx in range(n_layers):
            layer_fraction = fractions[..., layer_idx] * mask[layer_idx]
            material_fraction += layer_fraction
            if not np.any(layer_fraction > tol):
                continue

            n0, dn_c, dn_l = indices[layer_idx]
            eps0 = n0**2
            eps_c = 2.0 * n0 * dn_c
            eps_l = 2.0 * n0 * dn_l
            layer_m = self._tilted_layer_magnetization(
                tilted_magnetization,
                magnetization,
                layer_idx,
                (nz, ny, nx),
            )

            eps_iso += layer_fraction * eps0
            if abs(eps_c) > tol:
                gyrotropic_vector += (
                    layer_fraction[..., None] * eps_c * layer_m
                )
            if abs(eps_l) > tol:
                linear_tensor += (
                    layer_fraction[..., None, None]
                    * eps_l
                    * layer_m[..., :, None]
                    * layer_m[..., None, :]
                )

        vacuum_fraction = np.clip(1.0 - material_fraction, 0.0, 1.0)
        eps_iso += vacuum_fraction
        self._propagation_layer_thicknesses = thicknesses
        self.aperture_support_regions = None
        self.beam_direction = k
        return DynamicProjectedDielectricTensorStack(
            eps_iso=eps_iso,
            gyrotropic_vector=gyrotropic_vector,
            linear_tensor=linear_tensor,
            beam_direction=k,
        )

    def dielectric_tensor_mixed(self, n, theta=0.0):
        """Build a 3-element array of 2×2 transverse dielectric tensors.

        Uses the small-anisotropy approximation:
        ``eps0 = n0²``, ``eps_l = 2 n0 Δn_l``, ``eps_c = 2 n0 Δn_c``.

        Parameters
        ----------
        n : array-like of length 3
            ``(n0, dn_c, dn_l)`` where ``n0`` is the baseline isotropic
            refractive index, ``dn_c`` the circular anisotropy, and ``dn_l``
            the linear anisotropy contribution.
        theta : float, optional
            Rotation angle in radians of the linear principal axes.
            Default is ``0.0``.

        Returns
        -------
        ndarray of shape (3, 2, 2)
            ``[eps_isotropic * I, eps_circular_tensor, eps_linear_tensor]``
            where each element is a complex 2×2 matrix.
        """
        n0 = n[0]
        dn_c = n[1]
        dn_l = n[2]

        eps0 = n0**2
        eps_l = 2.0 * n0 * dn_l
        eps_c = 2.0 * n0 * dn_c

        # Linear anisotropy tensor in its own principal basis
        eps_l_tensor = np.array([[eps_l, 0.0], [0.0, -eps_l]], dtype=complex)

        # Rotate linear anisotropy tensor by theta
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]], dtype=complex)

        eps_l_rot = R @ eps_l_tensor @ R.T

        # Circular anisotropy tensor
        eps_c_tensor = np.array(
            [[0.0, 1.0j * eps_c], [-1.0j * eps_c, 0.0]], dtype=complex
        )

        eps_total = np.array([eps0 * np.eye(2, dtype=complex), eps_c_tensor, eps_l_rot])
        return eps_total

    def calc_effective_refractive_indices(
        self, refractive_index: complex, thickness: float
    ) -> complex:
        """Compute the effective refractive index for a single layer.

        Parameters
        ----------
        refractive_index : complex
            Complex refractive index of the layer material.
        thickness : float
            Physical thickness of the layer in metres.

        Returns
        -------
        complex
            Effective index ``refractive_index * thickness``.
        """
        return refractive_index * thickness

    def add_layer(self, element: str, thickness: float) -> None:
        """Append a material layer to the structure.

        Parameters
        ----------
        element : str
            Material name, must exist in ``self.material_params.database``.
        thickness : float
            Physical thickness of the layer in metres.
        thickness_nm : float
            Physical thickness of the layer in nanometres.

        Returns
        -------
        None
            The function completes in place.
        """
        refractive_index = self.material_params.get_refractive_index(element)
        dielectric_tensor = self.dielectric_tensor_mixed(n=refractive_index, theta=0.0)
        effective_index = self.calc_effective_refractive_indices(
            refractive_index, thickness
        )

        # NEED TO FIX THIS THICKNESS_NM BUSINESS

        self.layer_names.append(element)
        self.layer_thicknesses.append(thickness)
        self.layer_refractive_indices.append(refractive_index)
        self.dielectric_tensors.append(dielectric_tensor)
        self.effective_refractive_indices.append(effective_index)
        self.sample_shape[0] = len(self.layer_names)

    def add_effective_layer(
        self,
        label: str,
        components: tuple[tuple[str, float], ...],
    ) -> None:
        """Append one effective-medium layer from material/thickness components.

        The layer thickness is the sum of component thicknesses. Refractive
        indices and dielectric tensor channels are averaged by physical
        thickness, which preserves the existing isotropic, XMCD, and XMLD tensor
        representation while reducing the number of propagated slices.

        Parameters
        ----------
        label : str
            Input value for ``label``.
        components : tuple[tuple[str, float], ...]
            Input value for ``components``.

        Returns
        -------
        None
            The function completes in place.
        """
        if not components:
            raise ValueError("Composite layer must contain at least one component.")

        total_thickness = sum(thickness for _, thickness in components)
        if total_thickness <= 0:
            raise ValueError(
                f"Composite layer {label!r} must have positive total thickness."
            )

        refractive_index = np.zeros(3, dtype=complex)
        dielectric_tensor = None
        for element, thickness in components:
            weight = thickness / total_thickness
            component_index = np.asarray(
                self.material_params.get_refractive_index(element),
                dtype=complex,
            )
            component_tensor = self.dielectric_tensor_mixed(
                n=component_index,
                theta=0.0,
            )
            refractive_index += weight * component_index
            if dielectric_tensor is None:
                dielectric_tensor = weight * component_tensor
            else:
                dielectric_tensor += weight * component_tensor

        effective_index = self.calc_effective_refractive_indices(
            refractive_index,
            total_thickness,
        )

        self.layer_names.append(label)
        self.layer_thicknesses.append(total_thickness)
        self.layer_refractive_indices.append(refractive_index)
        self.dielectric_tensors.append(dielectric_tensor)
        self.effective_refractive_indices.append(effective_index)
        self.sample_shape[0] = len(self.layer_names)

    def return_layer_refractive_indices(self) -> NDArray[np.complex128]:
        """Return all layer refractive indices as a NumPy array.

        Returns
        -------
        NDArray[np.complex128]
            Array of shape ``(N,)`` with the complex refractive index of
            each layer in deposition order.

        Parameters
        ----------
        None
            This function takes no explicit input parameters.
        """
        return np.array(self.layer_refractive_indices)

    def return_layer_dielectric_tensors(self) -> NDArray[np.complex128]:
        """Return all layer dielectric tensors as a NumPy array.

        Returns
        -------
        NDArray[np.complex128]
            Array of shape ``(N,)`` with the complex dielectric tensors of
            each layer in deposition order.

        Parameters
        ----------
        None
            This function takes no explicit input parameters.
        """
        return np.array(self.dielectric_tensors)

    def remove_layer(self, index: int) -> None:
        """Remove a layer from the structure by its position index.

        Parameters
        ----------
        index : int
            Zero-based index of the layer to remove. Negative indices are
            supported (e.g. ``-1`` removes the last layer).

        Raises
        ------
        IndexError
            If ``index`` is out of range.

        Returns
        -------
        None
            The function completes in place.
        """
        n = len(self.layer_names)
        if index < -n or index >= n:
            raise IndexError(
                f"Layer index {index} out of range for structure with {n} layers."
            )
        self.layer_names.pop(index)
        self.layer_thicknesses.pop(index)
        self.layer_refractive_indices.pop(index)
        self.dielectric_tensors.pop(index)
        self.effective_refractive_indices.pop(index)

    def remove_layers_by_element(self, element: str) -> int:
        """Remove all layers of a given material from the structure.

        Parameters
        ----------
        element : str
            Material name whose layers should be removed.

        Returns
        -------
        int
            Number of layers removed.

        Raises
        ------
        ValueError
            If no layer with the given element name exists.
        """
        indices = [i for i, name in enumerate(self.layer_names) if name == element]
        if not indices:
            raise ValueError(f"No layers with element '{element}' found in structure.")
        for i in reversed(indices):
            self.layer_names.pop(i)
            self.layer_thicknesses.pop(i)
            self.layer_refractive_indices.pop(i)
            self.dielectric_tensors.pop(i)
            self.effective_refractive_indices.pop(i)
        return len(indices)

    def return_total_effective_refractive_index(self) -> complex:
        """Return the sum of effective refractive indices across all layers.

        Returns
        -------
        complex
            Sum of ``n * thickness`` over all layers.

        Parameters
        ----------
        None
            This function takes no explicit input parameters.
        """
        self.effective_refractive_index = sum(self.effective_refractive_indices)

        return self.effective_refractive_index

    def return_total_thickness(self) -> float:
        """Return the total physical thickness of the structure.

        Returns
        -------
        float
            Sum of layer thicknesses in metres.

        Parameters
        ----------
        None
            This function takes no explicit input parameters.
        """
        return sum(self.layer_thicknesses)

    def create_2d_refractive_index_map(self, shape: tuple) -> None:
        """Create a uniform 2-D effective refractive index map and store it.

        Fills ``self.refractive_index_map`` with a constant array equal to
        ``self.effective_refractive_index`` over the requested shape.
        :meth:`return_total_effective_refractive_index` must be called first.

        Parameters
        ----------
        shape : tuple of int
            Desired array shape ``(rows, cols)`` in pixels.

        Returns
        -------
        None
            The function completes in place.
        """
        n_layers = len(self.layer_refractive_indices)
        if n_layers == 0:
            raise ValueError(
                "Structure has no layers to create refractive index array."
            )

        self.refractive_index_map = (
            np.ones(shape, dtype=np.complex128) * self.effective_refractive_index
        )

    def return_total_effective_dielectric_tensor(self):
        """Return the thickness-weighted effective transverse dielectric tensor.

        Averages ``self.dielectric_tensors`` weighted by ``self.layer_thicknesses``.

        Returns
        -------
        ndarray of shape (3, 2, 2)
            Thickness-weighted mean dielectric tensor across all layers.

        Parameters
        ----------
        None
            This function takes no explicit input parameters.
        """

        D = np.sum(self.layer_thicknesses)
        eps_eff = np.zeros_like(np.asarray(self.dielectric_tensors[0], dtype=complex))

        for eps, d in zip(self.dielectric_tensors, self.layer_thicknesses):
            eps_eff += d * np.asarray(eps, dtype=complex)
        eps_eff /= D

        return eps_eff

    def calculate_final_dielectric_tensor(
        self,
        use_aperture_roi: bool = True,
        compact: bool = False,
        beam_direction: ArrayLike | None = None,
        beam_polarization_basis: tuple[ArrayLike, ArrayLike] | None = None,
        local_k_projection: bool = False,
    ) -> None:
        """Build the spatially resolved dielectric tensor for propagation.

        The base tensor is the charge/isotropic contribution for each layer.
        Magnetic XMCD and XMLD terms are only added where they can affect the
        hologram: inside the aperture support, i.e. pixels belonging to an
        OH/RH/slit opening in at least one layer. Intersecting aperture funnels
        are already combined by the multiplied mask and therefore become one
        connected support ROI. Outside that support the material remains
        diagonal, which lets the Jones propagator use its fast diagonal path for
        the gold-covered regions where domains are not visible.

        The 3-D ``mask`` still controls whether material or vacuum is present at
        each layer/pixel. Vacuum pixels are set to an identity dielectric tensor
        after the magnetic terms have been applied.

        Parameters
        ----------
        use_aperture_roi : bool
            If ``True``, split the aperture support into connected local
            bounding boxes and apply magnetic/vacuum corrections only in those
            regions. Physically intersecting funnels are one connected support
            and are not split into independent ROIs. If
            ``False``, use the full 2-D aperture-support mask, matching the
            pre-ROI optimization path for timing comparisons.

        compact : bool
            If ``True``, store a compact stack made of constant per-layer
            diagonal terms plus dense aperture ROI patches. This avoids
            allocating the full ``(Nz, Ny, Nx, 2, 2)`` tensor during multislice
            propagation.
        beam_direction : array-like of float, optional
            Beam propagation direction in lab ``(x, y, z)`` coordinates. When
            provided, the local 3-D magnetic dielectric response is projected
            onto two transverse Jones axes perpendicular to this direction.
            This produces a dense tensor stack and ignores ``compact``.
        beam_polarization_basis : pair of array-like, optional
            Optional lab-frame vectors defining the transverse Jones basis. The
            vectors are orthonormalized perpendicular to ``beam_direction``.
        local_k_projection : bool
            If ``True``, store direction-independent magnetic response channels
            and let the Jones propagator project each slice using the local
            phase-gradient direction of the current electric field. This makes
            XMCD proportional to the local ``m · k`` estimate. The tensor is
            dynamic and ignores ``compact``.

        Returns
        -------
        ndarray of shape (Nz,Ny,Nx, 2, 2)
            Final dielectric tensor stored in ``self.final_dielectric_tensor``.
        """

        if local_k_projection:
            self.final_dielectric_tensor = self._dynamic_projected_dielectric_tensor_stack(
                beam_direction=beam_direction,
            )
            return

        if beam_direction is not None:
            self.final_dielectric_tensor = self._dense_dielectric_tensor_for_beam_direction(
                beam_direction,
                beam_polarization_basis=beam_polarization_basis,
            )
            return

        use_tilted_voxelization = (
            not np.isclose(float(getattr(self, "sample_tilt_theta", 0.0)), 0.0)
            or getattr(self, "sample_tilt_simulation_z_extent", None) is not None
            or getattr(self, "tilted_magnetization", None) is not None
        )
        if use_tilted_voxelization:
            self.final_dielectric_tensor = self._tilted_dense_dielectric_tensor()
            return
        compact_stack = self.calculate_compact_dielectric_tensor(
            use_aperture_roi=use_aperture_roi
        )
        if compact:
            self.final_dielectric_tensor = compact_stack
            return
        self.final_dielectric_tensor = compact_stack.materialize()

    def calculate_final_scalar_refractive_index(
        self,
        pol,
        use_aperture_roi: bool = True,
        compact: bool = True,
        lazy: bool = True,
    ) -> None:
        """Build the spatially resolved scalar refractive-index stack.

        This is the scalar analogue of :meth:`calculate_final_dielectric_tensor`.
        Instead of constructing a ``(Nz, Ny, Nx, 2, 2)`` dielectric tensor stack,
        it combines the database refractive-index channels ``[n_total, n_circ,
        n_lin]`` with the selected polarization, magnetization, and aperture
        mask to create a complex ``(Nz, Ny, Nx)`` refractive-index stack.

        Parameters
        ----------
        pol : str or float
            Polarization used by the scalar eigenmode approximation.
        use_aperture_roi : bool
            If ``True``, store spatially varying magnetic/vacuum corrections in
            local aperture patches, mirroring the compact dielectric tensor
            representation.
        compact : bool
            If ``True``, store constant per-layer scalar indices plus aperture
            ROI patches. If ``False``, materialize the dense ``(Nz, Ny, Nx)``
            scalar index stack.
        lazy : bool
            If ``True`` and ``compact`` is also ``True``, compute scalar ROI
            patches layer by layer during propagation instead of precomputing
            all layer patches up front.
        """
        from scattering_calculator.beam_propagator import simple_propagation

        use_tilted_voxelization = (
            not np.isclose(float(getattr(self, "sample_tilt_theta", 0.0)), 0.0)
            or getattr(self, "sample_tilt_simulation_z_extent", None) is not None
            or getattr(self, "tilted_magnetization", None) is not None
        )
        if use_tilted_voxelization:
            self.final_scalar_refractive_index = self._tilted_dense_scalar_refractive_index(pol)
            return

        compact_stack = simple_propagation.calculate_scalar_refractive_index_stack(
            self,
            pol,
            use_aperture_roi=use_aperture_roi,
            lazy=lazy and compact,
        )
        if compact:
            self.final_scalar_refractive_index = compact_stack
            return
        self.final_scalar_refractive_index = compact_stack.materialize()

    def calculate_compact_dielectric_tensor(
        self,
        use_aperture_roi: bool = True,
    ) -> CompactDielectricTensorStack:
        """Build a compact dielectric tensor stack for lazy propagation.

        Parameters
        ----------
        use_aperture_roi : bool
            Input value for ``use_aperture_roi``.

        Returns
        -------
        result : CompactDielectricTensorStack
            Return value produced by the function.
        """

        mask = self.mask
        m = self.magnetization
        dt = self.dielectric_tensors
        if not isinstance(dt, np.ndarray):
            dt = np.asarray(dt)
            self.dielectric_tensors = dt

        eps0 = dt[:, 0]  # (Nz, 2, 2)
        eps_mz = dt[:, 1]  # (Nz, 2, 2)
        eps_xy = dt[:, 2]  # (Nz, 2, 2)

        base_diagonal = np.stack(
            (eps0[:, 0, 0], eps0[:, 1, 1]),
            axis=-1,
        ).astype(dt.dtype, copy=False)

        # Per-layer checks skip materials with no magnetic tensor contribution.
        tol = 1e-14
        has_mz = np.any(np.abs(eps_mz) > tol, axis=(1, 2))
        has_xy = np.any(np.abs(eps_xy) > tol, axis=(1, 2))

        # Open aperture support: magnetic contrast and vacuum corrections only
        # vary where the holography mask exposes holes. The ROI path splits that
        # support into local bounding boxes; the full-support path keeps the
        # earlier vectorised 2-D support mask for apples-to-apples benchmarks.
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
        self.aperture_support_regions = (
            support_objects if use_aperture_roi and has_aperture_support else None
        )

        def _region_active(layer_idx: int, region: tuple[slice, slice]) -> np.ndarray:
            """Handle the internal region active operation.

            Parameters
            ----------
            layer_idx : int
                Input value for ``layer_idx``.
            region : tuple[slice, slice]
                Input value for ``region``.

            Returns
            -------
            result : np.ndarray
                Return value produced by the function.
            """
            return aperture_support[region] & (mask[(layer_idx, *region)] > tol)

        patches: list[list[tuple[tuple[slice, slice], NDArray[np.complex128]]]] = [
            [] for _ in range(mask.shape[0])
        ]

        for layer_idx in range(mask.shape[0]):
            for region in support_objects:
                region_support = aperture_support[region]
                region_mask = mask[(layer_idx, *region)]
                needs_patch = np.any(region_support & (np.abs(region_mask - 1.0) > tol))
                if has_mz[layer_idx] or has_xy[layer_idx]:
                    needs_patch = needs_patch or np.any(_region_active(layer_idx, region))
                if not needs_patch:
                    continue

                region_shape = region_mask.shape
                eps_patch = np.zeros((*region_shape, 2, 2), dtype=dt.dtype)
                eps_patch[..., 0, 0] = base_diagonal[layer_idx, 0]
                eps_patch[..., 1, 1] = base_diagonal[layer_idx, 1]

                active_pixels = _region_active(layer_idx, region)
                layer_region = (layer_idx, *region)

                if has_mz[layer_idx] and np.any(active_pixels):
                    eps_patch[active_pixels] += (
                        m[layer_region + (2,)][active_pixels, None, None]
                        * eps_mz[layer_idx, None, :, :]
                    )

                if has_xy[layer_idx] and np.any(active_pixels):
                    mx = m[layer_region + (0,)][active_pixels]
                    my = m[layer_region + (1,)][active_pixels]
                    dxy = np.abs(mx) ** 2 - np.abs(my) ** 2
                    eps_patch[active_pixels] += (
                        dxy[:, None, None] * eps_xy[layer_idx, None, :, :]
                    )

                eps_patch *= region_mask[..., None, None]
                vac = 1.0 - region_mask
                eps_patch[..., 0, 0] += vac
                eps_patch[..., 1, 1] += vac
                patches[layer_idx].append((region, eps_patch))

        return CompactDielectricTensorStack(
            shape=(*mask.shape, 2, 2),
            base_diagonal=base_diagonal,
            patches=tuple(tuple(layer_patches) for layer_patches in patches),
            aperture_support_regions=(
                tuple(self.aperture_support_regions)
                if self.aperture_support_regions is not None
                else None
            ),
        )



    def calculate_final_dielectric_tensor_22052026(self) -> None:
        """Return the spatial-dependent dielectric tensor including XMCD and XMLD components
        multiplies the correct elements of the dielectric tensors with the correct components of the magnetization,
        to produce the magnetization dependent dielctric tensor.

        Some efficient safety checks save time by doing these operations only when necessary

        Returns
        -------
        ndarray of shape (Nz,Ny,Nx, 2, 2)
            Thickness-weighted mean dielectric tensor.

        Parameters
        ----------
        None
            This function takes no explicit input parameters.
        """

        mask = self.mask
        m = self.magnetization
        dt = self.dielectric_tensors
        if not isinstance(dt, np.ndarray):
            dt = np.asarray(dt)
            self.dielectric_tensors = dt

        eps0 = dt[:, 0]  # (Nz, 2, 2)
        eps_mz = dt[:, 1]  # (Nz, 2, 2)
        eps_xy = dt[:, 2]  # (Nz, 2, 2)

        out = np.empty((*mask.shape, 2, 2), dtype=dt.dtype)
        out[:] = eps0[:, None, None, :, :]

        # Cheap per-layer checks: shape (Nz,)
        tol = 1e-14
        has_mz = np.any(np.abs(eps_mz) > tol, axis=(1, 2))
        has_xy = np.any(np.abs(eps_xy) > tol, axis=(1, 2))

        if np.any(has_mz):
            out[has_mz] += (
                m[has_mz, ..., 2, None, None] * eps_mz[has_mz, None, None, :, :]
            )

        if np.any(has_xy):
            dxy = np.abs(m[..., 0]) ** 2 - np.abs(m[..., 1]) ** 2
            out[has_xy] += (
                dxy[has_xy, ..., None, None] * eps_xy[has_xy, None, None, :, :]
            )

        out *= mask[..., None, None]

        vac = 1.0 - mask
        out[..., 0, 0] += vac
        out[..., 1, 1] += vac

        self.final_dielectric_tensor = out

    def calculate_final_dielectric_tensor_old(self) -> None:
        """Run the calculate final dielectric tensor old operation.

        Parameters
        ----------
        None
            This function takes no explicit input parameters.

        Returns
        -------
        None
            The function completes in place.
        """
        mask = self.mask
        m = self.magnetization
        dt = self.dielectric_tensors
        if not isinstance(dt, np.ndarray):
            dt = np.asarray(dt)
            self.dielectric_tensors = dt

        eps0 = dt[:, 0]
        eps_mz = dt[:, 1]
        eps_xy = dt[:, 2]

        out = np.empty((*mask.shape, 2, 2), dtype=dt.dtype)
        out[:] = eps0[:, None, None, :, :]
        out += m[..., 2, None, None] * eps_mz[:, None, None, :, :]
        out += (np.abs(m[..., 0]) ** 2 - np.abs(m[..., 1]) ** 2)[
            ..., None, None
        ] * eps_xy[:, None, None, :, :]

        out *= mask[..., None, None]

        vac = 1.0 - mask
        out[..., 0, 0] += vac
        out[..., 1, 1] += vac

        self.final_dielectric_tensor = out

    def visualize_structure(self) -> None:
        """Plot the layer stack coloured by real and imaginary refractive index.

        Produces a side-by-side horizontal bar chart. The left panel shows the
        real part (refraction) and the right panel shows the imaginary part
        (absorption). Colour limits are normalised to the min/max across all
        layers. Each bar is labelled with the material name.

        Parameters
        ----------
        None
            This function takes no explicit input parameters.

        Returns
        -------
        None
            The function completes in place.
        """
        eps = 1e-12
        # Material entries contain [n0, dn_c, dn_l].  The stack overview uses
        # the isotropic n0 channel; plotting the magnetic contrast channels as
        # if each layer had one scalar index is both ambiguous and invalid.
        isotropic_indices = [
            complex(np.asarray(index).reshape(-1)[0])
            for index in self.layer_refractive_indices
        ]
        reals = [index.real for index in isotropic_indices]
        imags = [index.imag for index in isotropic_indices]
        norm_real = mcolors.Normalize(vmin=min(reals), vmax=max(reals) + eps)
        norm_imag = mcolors.Normalize(vmin=min(imags), vmax=max(imags) + eps)
        cmap = plt.cm.viridis_r

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        fig.suptitle(f"Structure: {self.name}")
        ax[0].set_title("Real (Refraction)")
        ax[1].set_title("Imaginary (Absorption)")

        y_bottom = 0
        for name, thickness, refractive_index in zip(
            self.layer_names, self.layer_thicknesses, isotropic_indices
        ):
            thickness = thickness * 1e9  # convert to nm for visualization
            y_center = y_bottom + thickness / 2
            for a, norm, val in [
                (ax[0], norm_real, refractive_index.real),
                (ax[1], norm_imag, refractive_index.imag),
            ]:
                a.barh(
                    y_center,
                    width=1,
                    height=thickness,
                    color=cmap(norm(val)),
                    edgecolor="black",
                )
                a.text(
                    0.5,
                    y_center,
                    name,
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="white",
                    fontweight="bold",
                )
            y_bottom += thickness

        fig.colorbar(
            plt.cm.ScalarMappable(norm=norm_real, cmap=cmap), ax=ax[0], label="Real(n)"
        )
        fig.colorbar(
            plt.cm.ScalarMappable(norm=norm_imag, cmap=cmap), ax=ax[1], label="Imag(n)"
        )
        ax[0].set_xlabel("Layer")
        ax[0].set_ylabel("Thickness (nm)")
        ax[1].set_xlabel("Layer")
        ax[1].set_ylabel("Thickness (nm)")
        plt.show()

    def calc_real_space_coordinates(self) -> None:
        """Compute real-space (x, y) coordinate grids for the illumination plane.

        Sets ``self.x`` and ``self.y`` as 2-D arrays of physical
        coordinates in metres, centred on the optical axis.

        Parameters
        ----------
        None
            This function takes no explicit input parameters.

        Returns
        -------
        None
            The function completes in place.
        """

        x = (
            np.arange(self.sample_shape[-1]) - self.sample_shape[-1] / 2
        ) * self.real_space_pixel_size
        y = (
            np.arange(self.sample_shape[-2]) - self.sample_shape[-2] / 2
        ) * self.real_space_pixel_size
        X, Y = np.meshgrid(x, y)
        self.x = X
        self.y = Y

    def get_extent_real_space(self) -> NDArray[np.float64]:
        """Calculate the physical extent of the sample plane in metres.

        Returns
        -------
        extent : tuple of float
            Physical size of the detector plane in metres as (min_x, max_x, min_y, max_y).

        Parameters
        ----------
        None
            This function takes no explicit input parameters.
        """

        self.extent_real = np.array(
            [
                np.min(self.x),
                np.max(self.x),
                np.min(self.y),
                np.max(self.y),
            ]
        )
        return self.extent_real


class Apertures2D:
    """2-D circular aperture mask generator.

    Parameters
    ----------
    shape : tuple of int
        Array shape ``(rows, cols)`` in pixels.
    real_space_pixel_size : float
        Physical size of one pixel in metres.

    Attributes
    ----------
    shape : tuple of int
        Array shape in pixels.
    pixel_size : float
        Physical pixel size in metres.
    aperture_design : ndarray of shape ``shape``
        Current aperture mask with values in ``[0, 1]``. Initialised to ones.
    x, y : ndarray
        2-D real-space coordinate grids in metres.
    extent_real : ndarray of shape (4,)
        ``[x_min, x_max, y_min, y_max]`` in metres.
    """

    def __init__(self, shape, real_space_pixel_size) -> None:
        """Initialize a Apertures2D instance.

        Parameters
        ----------
        shape : Any
            Input value for ``shape``.
        real_space_pixel_size : Any
            Input value for ``real_space_pixel_size``.

        Returns
        -------
        None
            The function completes in place.
        """
        self.shape = shape
        self.aperture_design = np.ones(shape)
        self.pixel_size = real_space_pixel_size

        self.calc_real_space_coordinates()
        self.extent_real = self.get_illumination_extent_real_space()

    def calc_real_space_coordinates(self) -> None:
        """Compute real-space (x, y) coordinate grids for the aperture.

        Sets ``self.x`` and ``self.y`` as 2-D arrays of physical
        coordinates in metres, centred on the optical axis.

        Parameters
        ----------
        None
            This function takes no explicit input parameters.

        Returns
        -------
        None
            The function completes in place.
        """

        x = (
            np.arange(self.sample_shape[1]) - self.sample_shape[1] / 2
        ) * self.pixel_size
        y = (
            np.arange(self.sample_shape[0]) - self.sample_shape[0] / 2
        ) * self.pixel_size
        X, Y = np.meshgrid(x, y)
        self.x = X
        self.y = Y

    def get_illumination_extent_real_space(self) -> NDArray[np.float64]:
        """Calculate the physical extent of the aperture plane in metres.

        Returns
        -------
        extent : tuple of float
            Physical size of the detector plane in metres as (min_x, max_x, min_y, max_y).

        Parameters
        ----------
        None
            This function takes no explicit input parameters.
        """

        self.extent_real = np.array(
            [
                np.min(self.x),
                np.max(self.x),
                np.min(self.y),
                np.max(self.y),
            ]
        )
        return self.extent_real

    def create_circle_aperture(
        self,
        center: tuple[float, float],
        radius: float,
        use_real_space_coordinates: bool = False,
        sigma: float | None = None,
    ) -> None:
        """Create a circular aperture mask and store it in ``self.aperture_design``.

        Parameters
        ----------
        center : tuple of int
            Mask centre coordinates (y, x) in pixels.
        radius : float
            Aperture radius in metres.
        use_real_space_coordinates : bool, optional
            If ``True``, convert the effective radius from metres to pixels
            using the real-space coordinate grid. Otherwise, treat the radius
            as already given in pixels.
        sigma : float or None, optional
            Edge transition width in pixels for the continuous aperture mask.
            A value of ``None`` or ``0`` gives a hard continuous boundary that
            is still averaged over each output pixel.

        Returns
        -------
        None
            The function completes in place.
        """
        if use_real_space_coordinates:
            # Convert radius from metres to pixels using the real-space grid
            pixel_radius = radius / np.abs(self.x[0, 1] - self.x[0, 0])
        else:
            pixel_radius = radius

        self.aperture_design = circle_mask(self.shape, center, pixel_radius, sigma)

    def return_aperture_mask(self) -> NDArray[np.float64]:
        """Return the current aperture design as a NumPy array.

        Returns
        -------
        NDArray[np.float64]
            2-D array of shape ``self.shape`` with values in [0, 1] representing
            the aperture mask.

        Parameters
        ----------
        None
            This function takes no explicit input parameters.
        """
        return self.aperture_design

    def visualize_aperture(self) -> None:
        """Display the current aperture design as an image.

        Parameters
        ----------
        None
            This function takes no explicit input parameters.

        Returns
        -------
        None
            The function completes in place.
        """
        fig, ax = plt.subplots(1, 2, figsize=(8, 4))
        ax[0].imshow(self.aperture_design)
        ax[0].set_title("Beamstop in px")
        ax[1].imshow(self.aperture_design, extent=1e6 * self.extent_real)
        ax[1].set_title("Beamstop in mm")
        ax[1].set_xlabel("x in µm")
        ax[1].set_ylabel("y in µm")


class Apertures3D:
    """3-D circular aperture mask generator for layered samples.

    Parameters
    ----------
    shape : tuple of int
        Array shape ``(layers, rows, cols)`` in pixels.
    real_space_pixel_size : float
        Physical size of one transverse pixel in metres.
    layer_thicknesses : array-like of float
        Physical thickness of each layer in metres.

    Attributes
    ----------
    shape : tuple of int
        Array shape ``(layers, rows, cols)`` in pixels.
    pixel_size : float
        Physical transverse pixel size in metres.
    layer_thicknesses : ndarray
        Physical thickness of each layer in metres.
    aperture_design : ndarray of shape ``shape``
        Current aperture mask with values in ``[0, 1]``. Initialised to ones.
    x, y, z : ndarray
        Sparse 3-D real-space coordinate grids in metres.
    extent_real : ndarray of shape (6,)
        ``[x_min, x_max, y_min, y_max, z_min, z_max]`` in metres.
    """

    def __init__(self, shape, real_space_pixel_size, layer_thicknesses) -> None:
        """Initialize a Apertures3D instance.

        Parameters
        ----------
        shape : Any
            Input value for ``shape``.
        real_space_pixel_size : Any
            Input value for ``real_space_pixel_size``.
        layer_thicknesses : Any
            Input value for ``layer_thicknesses``.

        Returns
        -------
        None
            The function completes in place.
        """
        self.shape = shape
        self.aperture_design = np.ones(shape)
        self.pixel_size = real_space_pixel_size
        self.layer_thicknesses = np.array(layer_thicknesses)

        self.calc_real_space_coordinates()
        self.extent_real = self.get_illumination_extent_real_space()

    def calc_real_space_coordinates(self) -> None:
        """Compute sparse real-space (x, y, z) coordinate grids.

        The sparse grids keep the old ``self.x[0, 1, 0]`` style indexing and
        extent calculations without allocating full ``(Ny, Nx, Nz)`` arrays.

        Parameters
        ----------
        None
            This function takes no explicit input parameters.

        Returns
        -------
        None
            The function completes in place.
        """

        x = (np.arange(self.shape[2]) - self.shape[2] / 2) * self.pixel_size
        y = (np.arange(self.shape[1]) - self.shape[1] / 2) * self.pixel_size
        z = np.cumsum(self.layer_thicknesses)
        X, Y, Z = np.meshgrid(x, y, z, indexing="xy", sparse=True)
        self.x = X
        self.y = Y
        self.z = Z

    def get_illumination_extent_real_space(self) -> NDArray[np.float64]:
        """Calculate the physical extent of the aperture plane in metres.

        Returns
        -------
        extent : tuple of float
            Physical size of the detector plane in metres as (min_x, max_x, min_y, max_y).

        Parameters
        ----------
        None
            This function takes no explicit input parameters.
        """

        self.extent_real = np.array(
            [
                np.min(self.x),
                np.max(self.x),
                np.min(self.y),
                np.max(self.y),
                np.min(self.z),
                np.max(self.z),
            ]
        )
        return self.extent_real

    def create_circle_aperture(
        self,
        center: tuple[float, float],
        radius: float,
        depth: float,
        use_real_space_coordinates: bool = False,
        sigma: float | None = None,
        angle: float = 0.0,
        ellipticity: float = 1.0,
        roughness: float = 0.0,
        roughness_modes: tuple[int, int] = (0, 0),
        seed: int | None = None,
        use_roi: bool = True,
        top_radius_factor: float = 2.0,
        taper_depth: float | None = None,
    ) -> None:
        """Create a tapered circular aperture mask.

        The supplied ``radius`` is the base radius at the bottom of the drilled
        depth. ``top_radius_factor`` controls the top opening radius; the
        radius changes linearly down to ``radius`` over ``taper_depth``. Layers
        deeper than ``taper_depth`` keep the base radius, giving a cylindrical
        lower aperture.

        Parameters
        ----------
        center : tuple of int
            Mask centre coordinates (y, x) in pixels.
        radius : float
            Aperture radius in metres.
        depth : float
            depth of the aperture in metres
        use_real_space_coordinates : bool, optional
            If ``True``, convert the effective radius from metres to pixels
            using the real-space coordinate grid. Otherwise, treat the radius
            as already given in pixels.
        sigma : float or None, optional
            Edge transition width in pixels for the continuous aperture mask.
            A value of ``None`` or ``0`` gives a hard continuous boundary that
            is still averaged over each output pixel.
        top_radius_factor : float, optional
            Ratio between the aperture radius at the top surface and the base
            radius at ``depth``. ``1`` gives the old cylindrical aperture;
            ``2`` gives a cone whose top opening is twice the base radius.
        taper_depth : float or None, optional
            Physical depth over which the aperture tapers from the top radius
            to the base radius. If ``None``, the taper spans the full drilled
            depth, matching the original conical behaviour.

        Returns
        -------
        None
            The function completes in place.
        """
        top_radius_factor = float(top_radius_factor)
        if top_radius_factor <= 0:
            raise ValueError(
                f"top_radius_factor must be positive, got {top_radius_factor}"
            )

        if use_real_space_coordinates:
            # Convert radius from metres to pixels using the real-space grid
            pixel_radius = radius / np.abs(self.x[0, 1, 0] - self.x[0, 0, 0])
            pixel_depth = np.argmin(
                np.abs(np.append(0, np.cumsum(self.layer_thicknesses)) - depth)
            )
            pixel_sigma = (
                None
                if sigma is None
                else sigma / np.abs(self.x[0, 1, 0] - self.x[0, 0, 0])
            )
            pixel_center = np.array(center) / self.pixel_size + self.shape[1] // 2
            pixel_taper_depth = None if taper_depth is None else float(taper_depth)
        else:
            pixel_radius = radius
            pixel_depth = depth
            pixel_sigma = sigma
            pixel_center = np.array(center)
            pixel_taper_depth = None if taper_depth is None else float(taper_depth)
        pixel_depth = int(np.clip(pixel_depth, 0, self.shape[0]))
        if pixel_depth <= 0:
            return

        max_pixel_radius = pixel_radius * max(1.0, top_radius_factor)
        if use_roi:
            y_slice, x_slice = self._aperture_bbox(
                self.shape,
                pixel_center,
                max_pixel_radius,
                sigma=pixel_sigma,
                angle=angle,
                ellipticity=ellipticity,
                roughness=roughness,
            )
            local_shape = (
                self.shape[0],
                y_slice.stop - y_slice.start,
                x_slice.stop - x_slice.start,
            )
            local_center = (
                pixel_center[0] - y_slice.start,
                pixel_center[1] - x_slice.start,
            )
        else:
            y_slice = slice(0, self.shape[1])
            x_slice = slice(0, self.shape[2])
            local_shape = self.shape
            local_center = pixel_center

        layer_edges = np.concatenate(([0.0], np.cumsum(self.layer_thicknesses)))
        if use_real_space_coordinates:
            drilled_depth = float(layer_edges[pixel_depth])
            layer_top_depths = layer_edges[:pixel_depth]
            taper_limit = drilled_depth if pixel_taper_depth is None else min(
                max(pixel_taper_depth, 0.0), drilled_depth
            )
            if taper_limit > 0:
                depth_fraction = np.clip(layer_top_depths / taper_limit, 0.0, 1.0)
            else:
                depth_fraction = np.ones(pixel_depth, dtype=float)
        else:
            layer_top_indices = np.arange(0, pixel_depth, dtype=float)
            taper_limit = (
                float(pixel_depth)
                if pixel_taper_depth is None
                else min(max(pixel_taper_depth, 0.0), float(pixel_depth))
            )
            if taper_limit > 0:
                depth_fraction = np.clip(layer_top_indices / taper_limit, 0.0, 1.0)
            else:
                depth_fraction = np.ones(pixel_depth, dtype=float)
        layer_radii = pixel_radius * (
            top_radius_factor + (1.0 - top_radius_factor) * depth_fraction
        )
        for layer_idx, layer_radius in enumerate(layer_radii):
            layer_seed = None if seed is None else int(seed) + 104729 * layer_idx
            hole_mask = self._aperture_hole_mask(
                local_shape,
                local_center,
                layer_radius,
                pixel_sigma,
                angle=angle,
                ellipticity=ellipticity,
                roughness=roughness,
                roughness_modes=roughness_modes,
                seed=layer_seed,
            )
            self.aperture_design[layer_idx, y_slice, x_slice] *= 1 - hole_mask

    def create_slit_aperture(
        self,
        center: tuple[float, float],
        width: float,
        length: float,
        depth: float,
        use_real_space_coordinates: bool = False,
        sigma: float | None = None,
        angle: float = 0.0,
        roughness: float = 0.0,
        roughness_modes: tuple[int, int] = (0, 0),
        seed: int | None = None,
        use_roi: bool = True,
        top_radius_factor: float = 2.0,
        taper_depth: float | None = None,
    ) -> None:
        """Create a tapered rectangular slit aperture mask.

        ``width`` is the short side and reuses the aperture radius list in the
        pipeline configuration. ``length`` is the long side. Both dimensions are
        scaled by ``top_radius_factor`` at the top surface and taper to the base
        dimensions over ``taper_depth``.
        """
        top_radius_factor = float(top_radius_factor)
        if top_radius_factor <= 0:
            raise ValueError(
                f"top_radius_factor must be positive, got {top_radius_factor}"
            )

        if use_real_space_coordinates:
            pixel_size = np.abs(self.x[0, 1, 0] - self.x[0, 0, 0])
            pixel_width = width / pixel_size
            pixel_length = length / pixel_size
            pixel_depth = np.argmin(
                np.abs(np.append(0, np.cumsum(self.layer_thicknesses)) - depth)
            )
            pixel_sigma = None if sigma is None else sigma / pixel_size
            pixel_center = np.array(center) / self.pixel_size + self.shape[1] // 2
            pixel_taper_depth = None if taper_depth is None else float(taper_depth)
        else:
            pixel_width = width
            pixel_length = length
            pixel_depth = depth
            pixel_sigma = sigma
            pixel_center = np.array(center)
            pixel_taper_depth = None if taper_depth is None else float(taper_depth)
        pixel_depth = int(np.clip(pixel_depth, 0, self.shape[0]))
        if pixel_depth <= 0:
            return

        max_width = pixel_width * max(1.0, top_radius_factor)
        max_length = pixel_length * max(1.0, top_radius_factor)
        if use_roi:
            y_slice, x_slice = self._rectangle_aperture_bbox(
                self.shape,
                pixel_center,
                max_width,
                max_length,
                sigma=pixel_sigma,
                angle=angle,
                roughness=roughness,
            )
            local_shape = (
                self.shape[0],
                y_slice.stop - y_slice.start,
                x_slice.stop - x_slice.start,
            )
            local_center = (
                pixel_center[0] - y_slice.start,
                pixel_center[1] - x_slice.start,
            )
        else:
            y_slice = slice(0, self.shape[1])
            x_slice = slice(0, self.shape[2])
            local_shape = self.shape
            local_center = pixel_center

        layer_edges = np.concatenate(([0.0], np.cumsum(self.layer_thicknesses)))
        if use_real_space_coordinates:
            drilled_depth = float(layer_edges[pixel_depth])
            layer_top_depths = layer_edges[:pixel_depth]
            taper_limit = drilled_depth if pixel_taper_depth is None else min(
                max(pixel_taper_depth, 0.0), drilled_depth
            )
            depth_fraction = (
                np.clip(layer_top_depths / taper_limit, 0.0, 1.0)
                if taper_limit > 0
                else np.ones(pixel_depth, dtype=float)
            )
        else:
            layer_top_indices = np.arange(0, pixel_depth, dtype=float)
            taper_limit = (
                float(pixel_depth)
                if pixel_taper_depth is None
                else min(max(pixel_taper_depth, 0.0), float(pixel_depth))
            )
            depth_fraction = (
                np.clip(layer_top_indices / taper_limit, 0.0, 1.0)
                if taper_limit > 0
                else np.ones(pixel_depth, dtype=float)
            )
        layer_widths = pixel_width * (
            top_radius_factor + (1.0 - top_radius_factor) * depth_fraction
        )
        layer_lengths = pixel_length * (
            top_radius_factor + (1.0 - top_radius_factor) * depth_fraction
        )
        for layer_idx, (layer_width, layer_length) in enumerate(
            zip(layer_widths, layer_lengths)
        ):
            layer_seed = None if seed is None else int(seed) + 104729 * layer_idx
            hole_mask = self._rectangle_aperture_hole_mask(
                local_shape,
                local_center,
                layer_width,
                layer_length,
                pixel_sigma,
                angle=angle,
                roughness=roughness,
                roughness_modes=roughness_modes,
                seed=layer_seed,
            )
            self.aperture_design[layer_idx, y_slice, x_slice] *= 1 - hole_mask

    @staticmethod
    def _aperture_bbox(
        shape,
        center,
        radius,
        sigma=None,
        angle: float = 0.0,
        ellipticity: float = 1.0,
        roughness: float = 0.0,
    ) -> tuple[slice, slice]:
        """Return a tight y/x bounding box for an aperture hole.

        Parameters
        ----------
        shape : Any
            Input value for ``shape``.
        center : Any
            Input value for ``center``.
        radius : Any
            Input value for ``radius``.
        sigma : Any
            Input value for ``sigma``.
        angle : float
            Input value for ``angle``.
        ellipticity : float
            Input value for ``ellipticity``.
        roughness : float
            Input value for ``roughness``.

        Returns
        -------
        result : tuple[slice, slice]
            Return value produced by the function.
        """
        _, ny, nx = shape
        ellipticity = float(ellipticity)
        if ellipticity <= 0:
            raise ValueError(f"ellipticity must be positive, got {ellipticity}")

        radius_y = max(float(radius) * np.sqrt(ellipticity), 1e-12)
        radius_x = max(float(radius) / np.sqrt(ellipticity), 1e-12)
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        half_x = np.sqrt((radius_x * cos_angle) ** 2 + (radius_y * sin_angle) ** 2)
        half_y = np.sqrt((radius_x * sin_angle) ** 2 + (radius_y * cos_angle) ** 2)
        roughness_scale = 1.0 + max(0.0, 2.0 * float(roughness))
        sigma_pad = 0.0 if sigma is None else 4.0 * abs(float(sigma))
        half_x = half_x * roughness_scale + sigma_pad + 2.0
        half_y = half_y * roughness_scale + sigma_pad + 2.0

        y0 = max(0, int(np.floor(center[0] - half_y)))
        y1 = min(ny, int(np.ceil(center[0] + half_y)) + 1)
        x0 = max(0, int(np.floor(center[1] - half_x)))
        x1 = min(nx, int(np.ceil(center[1] + half_x)) + 1)
        return slice(y0, y1), slice(x0, x1)

    @staticmethod
    def _rectangle_aperture_bbox(
        shape,
        center,
        width,
        length,
        sigma=None,
        angle: float = 0.0,
        roughness: float = 0.0,
    ) -> tuple[slice, slice]:
        """Return a tight y/x bounding box for a rectangular slit."""
        _, ny, nx = shape
        half_width = max(float(width) / 2.0, 1e-12)
        half_length = max(float(length) / 2.0, 1e-12)
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        half_x = abs(half_length * cos_angle) + abs(half_width * sin_angle)
        half_y = abs(half_length * sin_angle) + abs(half_width * cos_angle)
        roughness_pad = max(half_width, half_length) * max(0.0, 2.0 * float(roughness))
        sigma_pad = 0.0 if sigma is None else 4.0 * abs(float(sigma))
        half_x = half_x + roughness_pad + sigma_pad + 2.0
        half_y = half_y + roughness_pad + sigma_pad + 2.0

        y0 = max(0, int(np.floor(center[0] - half_y)))
        y1 = min(ny, int(np.ceil(center[0] + half_y)) + 1)
        x0 = max(0, int(np.floor(center[1] - half_x)))
        x1 = min(nx, int(np.ceil(center[1] + half_x)) + 1)
        return slice(y0, y1), slice(x0, x1)

    @staticmethod
    def _aperture_hole_mask(
        shape,
        center,
        radius,
        sigma=None,
        angle: float = 0.0,
        ellipticity: float = 1.0,
        roughness: float = 0.0,
        roughness_modes: tuple[int, int] = (0, 0),
        seed: int | None = None,
    ) -> NDArray[np.float64]:
        """Create a possibly elliptical and rough aperture-hole mask.

        Pixel values are estimated from the signed distance to the continuous
        aperture boundary on the native grid. This gives fractional edge pixels
        and keeps subpixel holes from snapping to binary full-open/full-closed
        pixels without repeatedly supersampling the aperture crop.

        Parameters
        ----------
        shape : Any
            Input value for ``shape``.
        center : Any
            Input value for ``center``.
        radius : Any
            Input value for ``radius``.
        sigma : Any
            Input value for ``sigma``.
        angle : float
            Input value for ``angle``.
        ellipticity : float
            Input value for ``ellipticity``.
        roughness : float
            Input value for ``roughness``.
        roughness_modes : tuple[int, int]
            Input value for ``roughness_modes``.
        seed : int | None
            Input value for ``seed``.

        Returns
        -------
        result : NDArray[np.float64]
            Return value produced by the function.
        """
        ellipticity = float(ellipticity)
        if ellipticity <= 0:
            raise ValueError(f"ellipticity must be positive, got {ellipticity}")

        _, ny, nx = shape
        radius_y = radius * np.sqrt(ellipticity)
        radius_x = radius / np.sqrt(ellipticity)
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)

        rough_modes = []
        if roughness > 0:
            rng = np.random.default_rng(seed)
            min_mode, max_mode = roughness_modes
            for mode in range(max(1, int(min_mode)), int(max_mode) + 1):
                rough_modes.append(
                    (
                        mode,
                        float(rng.normal(scale=roughness / mode)),
                        float(rng.uniform(0.0, 2.0 * np.pi)),
                    )
                )

        dy = np.arange(ny, dtype=float)[:, None] - center[0]
        dx = np.arange(nx, dtype=float)[None, :] - center[1]
        xr = cos_angle * dx + sin_angle * dy
        yr = -sin_angle * dx + cos_angle * dy
        normalized_radius = np.sqrt((xr / radius_x) ** 2 + (yr / radius_y) ** 2)

        boundary = 1.0
        if rough_modes:
            polar_angle = np.arctan2(yr / radius_y, xr / radius_x)
            boundary = np.ones((ny, nx), dtype=float)
            for mode, amplitude, phase in rough_modes:
                boundary += amplitude * np.cos(mode * polar_angle + phase)
            boundary = np.clip(
                boundary, 1.0 - 2.0 * roughness, 1.0 + 2.0 * roughness
            )

        effective_radius = np.sqrt(radius_y * radius_x)
        signed_distance = (boundary - normalized_radius) * effective_radius
        edge_width = max(0.0, float(sigma or 0.0))
        if edge_width > 0:
            # Combine the physical edge transition with the native-pixel
            # footprint. This is a deterministic, single-pass approximation to
            # pixel-area averaging, not a supersampled rasterization.
            transition_width = edge_width + 0.5
            transition = np.clip(
                0.5 + 0.5 * signed_distance / transition_width,
                0.0,
                1.0,
            )
            mask = transition * transition * (3.0 - 2.0 * transition)
        else:
            mask = np.clip(signed_distance + 0.5, 0.0, 1.0)

        if max(radius_y, radius_x) < 1.5:
            rough_area_factor = 1.0 + 0.5 * sum(
                amplitude**2 for _, amplitude, _ in rough_modes
            )
            target_area = np.pi * radius_y * radius_x * rough_area_factor
            current_area = np.sum(mask)
            if current_area > 0:
                mask *= min(1.0, target_area / current_area)

        return np.clip(mask, 0.0, 1.0)

    @staticmethod
    def _rectangle_aperture_hole_mask(
        shape,
        center,
        width,
        length,
        sigma=None,
        angle: float = 0.0,
        roughness: float = 0.0,
        roughness_modes: tuple[int, int] = (0, 0),
        seed: int | None = None,
    ) -> NDArray[np.float64]:
        """Create a rectangular slit aperture-hole mask on the native grid."""
        _, ny, nx = shape
        half_width = max(float(width) / 2.0, 1e-12)
        half_length = max(float(length) / 2.0, 1e-12)
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)

        rough_modes = []
        if roughness > 0:
            rng = np.random.default_rng(seed)
            min_mode, max_mode = roughness_modes
            for mode in range(max(1, int(min_mode)), int(max_mode) + 1):
                rough_modes.append(
                    (
                        mode,
                        float(rng.normal(scale=roughness / mode)),
                        float(rng.uniform(0.0, 2.0 * np.pi)),
                    )
                )

        dy = np.arange(ny, dtype=float)[:, None] - center[0]
        dx = np.arange(nx, dtype=float)[None, :] - center[1]
        xr = cos_angle * dx + sin_angle * dy
        yr = -sin_angle * dx + cos_angle * dy

        qx = np.abs(xr) - half_length
        qy = np.abs(yr) - half_width
        outside = np.sqrt(np.maximum(qx, 0.0) ** 2 + np.maximum(qy, 0.0) ** 2)
        inside = np.minimum(np.maximum(qx, qy), 0.0)
        signed_distance = -(outside + inside)

        if rough_modes:
            polar_angle = np.arctan2(yr / half_width, xr / half_length)
            rough_offset = np.zeros((ny, nx), dtype=float)
            rough_scale = min(half_width, half_length)
            for mode, amplitude, phase in rough_modes:
                rough_offset += amplitude * np.cos(mode * polar_angle + phase)
            rough_offset = np.clip(
                rough_offset, -2.0 * roughness, 2.0 * roughness
            )
            signed_distance += rough_offset * rough_scale

        edge_width = max(0.0, float(sigma or 0.0))
        if edge_width > 0:
            transition_width = edge_width + 0.5
            transition = np.clip(
                0.5 + 0.5 * signed_distance / transition_width,
                0.0,
                1.0,
            )
            mask = transition * transition * (3.0 - 2.0 * transition)
        else:
            mask = np.clip(signed_distance + 0.5, 0.0, 1.0)

        if max(half_width, half_length) < 1.5:
            target_area = width * length
            current_area = np.sum(mask)
            if current_area > 0:
                mask *= min(1.0, target_area / current_area)

        return np.clip(mask, 0.0, 1.0)

    def create_empty_aperture(self) -> None:
        """Create an empty aperture mask (all zeros) and store it in ``self.aperture_design``.

        Parameters
        ----------
        None
            This function takes no explicit input parameters.

        Returns
        -------
        None
            The function completes in place.
        """
        self.aperture_design = np.zeros(self.shape)

    def return_aperture_mask(self) -> NDArray[np.float64]:
        """Return the current aperture design as a NumPy array.

        Returns
        -------
        NDArray[np.float64]
            2-D array of shape ``self.shape`` with values in [0, 1] representing
            the aperture mask.

        Parameters
        ----------
        None
            This function takes no explicit input parameters.
        """
        return self.aperture_design

    def visualize_aperture(self) -> None:
        """Display the current aperture design as an image.

        Parameters
        ----------
        None
            This function takes no explicit input parameters.

        Returns
        -------
        None
            The function completes in place.
        """
        fig, ax = plt.subplots(1, 2, figsize=(8, 4))
        ax[0].imshow(np.average(self.aperture_design, axis=0))
        ax[0].set_title("Beamstop in px")
        ax[1].imshow(
            np.average(self.aperture_design, axis=0), extent=1e6 * self.extent_real
        )
        ax[1].set_title("Beamstop in mm")
        ax[1].set_xlabel("x in µm")
        ax[1].set_ylabel("y in µm")


class Magnetic_Structure:
    """Extension of the Structure class to include magnetic properties.

    Parameters
    ----------
    name : str
        Human-readable label for the structure.
    material_params : material_params
        Material database providing refractive indices for each element.

    Attributes
    ----------
    name : str
        Structure label.
    material_params : material_params
        Reference to the material database.
    layer_names : list of str
        Element name of each layer in deposition order.
    layer_thicknesses : list of float
        Physical thickness of each layer in metres.
    layer_refractive_indices : list of complex
        Complex refractive index of each layer.
    effective_refractive_indices : list of complex
        Effective index (``n * thickness``) of each layer.
    magnetization_profiles : list of complex
        Complex magnetization profile for each layer, representing the magnetic contribution to the refractive index.
    """

    def __init__(self, magnetic_structure, magnetic_pattern) -> None:
        """Parameters
        ----------
        magnetic_structure : Structure
            Fully assembled :class:`Structure` whose
            ``effective_refractive_index`` carries the magnetic contribution.
        magnetic_pattern : ndarray
            2-D (or 3-D) magnetisation pattern with values in ``[-1, 1]``.
            A 3-D array is interpreted as ``(depth, rows, cols)``.

        Returns
        -------
        None
            The function completes in place.
        """
        self.magnetic_structure = magnetic_structure
        self.magnetic_refractive_index = magnetic_structure.effective_refractive_index
        self.magnetic_pattern = magnetic_pattern

    def calc_projection_approximation(self) -> None:
        """Compute the projection approximation of the magnetisation pattern.

        For a 3-D pattern ``(depth, rows, cols)``, sums along the depth axis
        to obtain a 2-D projected map. For a 2-D pattern the input is used
        directly. The result is stored in ``self.magnetic_projection``.

        Parameters
        ----------
        None
            This function takes no explicit input parameters.

        Returns
        -------
        None
            The function completes in place.
        """
        if self.magnetic_pattern.ndim > 2:
            self.magnetic_projection = np.sum(self.magnetic_pattern, axis=0)
        else:
            self.magnetic_projection = self.magnetic_pattern

    def visualize_magnetic_projection(self) -> None:
        """Visualize the magnetic projection approximation.

        Parameters
        ----------
        None
            This function takes no explicit input parameters.

        Returns
        -------
        None
            The function completes in place.
        """
        if hasattr(self, "magnetic_projection"):
            _, ax = plt.subplots()
            m = ax.imshow(self.magnetic_projection, cmap="gray")
            plt.colorbar(m, label="Magnetic Projection")
            ax.set_title("Projection Approximation of Magnetic Pattern")
            ax.set_xlabel("X (pixels)")
            ax.set_ylabel("Y (pixels)")
            plt.show()
        else:
            print(
                "Magnetic projection not calculated yet. Call calc_projection_approximation() first."
            )

    def calc_magnetic_dichroism_birefringence(self) -> None:
        """Compute spatially resolved magnetic dichroism and birefringence maps.

        Multiplies the effective magnetic refractive index by the magnetisation
        pattern to obtain per-pixel dichroism (imaginary part) and birefringence
        (real part) maps. Results are stored in:

        - ``self.magnetic_dichroism``
        - ``self.magnetic_birefringence``
        - ``self.magnetic_refractive_index_map``

        Parameters
        ----------
        None
            This function takes no explicit input parameters.

        Returns
        -------
        None
            The function completes in place.
        """
        self.magnetic_dichroism = (
            self.magnetic_refractive_index.imag * self.magnetic_pattern
        )
        self.magnetic_birefringence = (
            self.magnetic_refractive_index.real * self.magnetic_pattern
        )
        self.magnetic_refractive_index_map = (
            self.magnetic_dichroism + 1j * self.magnetic_birefringence
        )

    def visualize_magnetic_contributions(self):
        """Visualize the magnetic dichroism and birefringence contributions.

        Parameters
        ----------
        None
            This function takes no explicit input parameters.

        Returns
        -------
        result : Any
            Return value produced by the function.
        """
        if hasattr(self, "magnetic_dichroism") and hasattr(
            self, "magnetic_birefringence"
        ):
            vmin, vmax = np.min(
                [self.magnetic_dichroism, self.magnetic_birefringence]
            ), np.max([self.magnetic_dichroism, self.magnetic_birefringence])

            fig, ax = plt.subplots(1, 2, figsize=(11, 4))
            im1 = ax[0].imshow(self.magnetic_dichroism, vmin=vmin, vmax=vmax)
            fig.colorbar(im1, ax=ax[0], label="Magnetic Dichroism (a.u.)")
            ax[0].set_title("Magnetic Dichroism Contribution")
            ax[0].set_xlabel("X (pixels)")
            ax[0].set_ylabel("Y (pixels)")

            im2 = ax[1].imshow(self.magnetic_birefringence, vmin=vmin, vmax=vmax)
            fig.colorbar(im2, ax=ax[1], label="Magnetic Birefringence (a.u.)")
            ax[1].set_title("Magnetic Birefringence Contribution")
            ax[1].set_xlabel("X (pixels)")
            ax[1].set_ylabel("Y (pixels)")
            plt.show()
        else:
            print(
                "Magnetic contributions not calculated yet. Call calc_magnetic_dichroism_birefringence() first."
            )

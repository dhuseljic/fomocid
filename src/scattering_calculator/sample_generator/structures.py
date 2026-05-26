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
from typing import List, Tuple


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
        if self.components is None:
            object.__setattr__(
                self,
                "components",
                ((self.material, self.thickness),),
            )

    @property
    def is_composite(self) -> bool:
        """Return ``True`` when this layer combines multiple materials."""
        return len(self.components or ()) > 1

    # @property
    # def thickness(self) -> float:
    #    return self.thickness_nm * 1e-9

    def to_txt_line(self) -> str:
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
        """Total physical thickness of the stack in metres."""
        return sum(layer.thickness for layer in self.layers)

    def add_comment(self, text: str) -> None:
        """Append a free-text comment to the recipe.

        Parameters
        ----------
        text : str
            Comment string to append.
        """
        self.comments.append(text)

    def summary(self) -> str:
        """Return a human-readable summary of the recipe.

        Returns
        -------
        str
            Multi-line string listing sample name, recipe, layer count,
            total thickness, and any attached comments.
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
    """
    Convert thickness in nm to a compact string in meters.
    Examples:
        5    -> '5e-9'
        1.5  -> '1.5e-9'
        0.25 -> '0.25e-9'
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
        """
        layers = self._parse_sequence(stop_char=None)
        if self.pos != len(self.text):
            raise ValueError(
                f"Unexpected trailing content at position {self.pos}: "
                f"{self.text[self.pos:]}"
            )
        return layers

    def _parse_sequence(self, stop_char: str | None) -> List[Layer]:
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
        self._expect("[")
        inner_layers = self._parse_sequence(stop_char="]")
        self._expect("]")

        self._expect("x")
        repeat = self._parse_integer()

        return inner_layers * repeat

    def _parse_composite_layer(self) -> Layer:
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

        if self.sample_shape[0] == 0:
            self.sample_shape[0] = len(self.layer_names)

        self.calc_real_space_coordinates()
        self.get_extent_real_space()

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
        """
        return np.array(self.layer_refractive_indices)

    def return_layer_dielectric_tensors(self) -> NDArray[np.complex128]:
        """Return all layer dielectric tensors as a NumPy array.

        Returns
        -------
        NDArray[np.complex128]
            Array of shape ``(N,)`` with the complex dielectric tensors of
            each layer in deposition order.
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
        """
        self.effective_refractive_index = sum(self.effective_refractive_indices)

        return self.effective_refractive_index

    def return_total_thickness(self) -> float:
        """Return the total physical thickness of the structure.

        Returns
        -------
        float
            Sum of layer thicknesses in metres.
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
        """
        Return the thickness-weighted effective transverse dielectric tensor.

        Averages ``self.dielectric_tensors`` weighted by ``self.layer_thicknesses``.

        Returns
        -------
        ndarray of shape (3, 2, 2)
            Thickness-weighted mean dielectric tensor across all layers.
        """

        D = np.sum(self.layer_thicknesses)
        eps_eff = np.zeros_like(np.asarray(self.dielectric_tensors[0], dtype=complex))

        for eps, d in zip(self.dielectric_tensors, self.layer_thicknesses):
            eps_eff += d * np.asarray(eps, dtype=complex)
        eps_eff /= D

        return eps_eff

    def calculate_final_dielectric_tensor(self, use_aperture_roi: bool = True) -> None:
        """Build the spatially resolved dielectric tensor for propagation.

        The base tensor is the charge/isotropic contribution for each layer.
        Magnetic XMCD and XMLD terms are only added where they can affect the
        hologram: inside the aperture support, i.e. pixels belonging to an OH/RH
        opening in at least one layer. Outside that support the material remains
        diagonal, which lets the Jones propagator use its fast diagonal path for
        the gold-covered regions where domains are not visible.

        The 3-D ``mask`` still controls whether material or vacuum is present at
        each layer/pixel. Vacuum pixels are set to an identity dielectric tensor
        after the magnetic terms have been applied.

        Parameters
        ----------
        use_aperture_roi : bool
            If ``True``, split the aperture support into local bounding boxes and
            apply magnetic/vacuum corrections only in those regions. If
            ``False``, use the full 2-D aperture-support mask, matching the
            pre-ROI optimization path for timing comparisons.

        Returns
        -------
        ndarray of shape (Nz,Ny,Nx, 2, 2)
            Final dielectric tensor stored in ``self.final_dielectric_tensor``.
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

        out = np.zeros((*mask.shape, 2, 2), dtype=dt.dtype)
        out[..., 0, 0] = eps0[:, None, None, 0, 0]
        out[..., 1, 1] = eps0[:, None, None, 1, 1]

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
            return aperture_support[region] & (mask[(layer_idx, *region)] > tol)

        for layer_idx in np.flatnonzero(has_mz):
            for region in support_objects:
                active_pixels = _region_active(layer_idx, region)
                if not np.any(active_pixels):
                    continue
                layer_region = (layer_idx, *region)
                out[layer_region][active_pixels] += (
                    m[layer_region + (2,)][active_pixels, None, None]
                    * eps_mz[layer_idx, None, :, :]
                )

        for layer_idx in np.flatnonzero(has_xy):
            for region in support_objects:
                active_pixels = _region_active(layer_idx, region)
                if not np.any(active_pixels):
                    continue
                layer_region = (layer_idx, *region)
                mx = m[layer_region + (0,)][active_pixels]
                my = m[layer_region + (1,)][active_pixels]
                dxy = np.abs(mx) ** 2 - np.abs(my) ** 2
                out[layer_region][active_pixels] += (
                    dxy[:, None, None] * eps_xy[layer_idx, None, :, :]
                )

        for region in support_objects:
            region_mask = mask[(slice(None), *region)]
            out_region = out[(slice(None), *region)]
            out_region *= region_mask[..., None, None]
            vac = 1.0 - region_mask
            out_region[..., 0, 0] += vac
            out_region[..., 1, 1] += vac

        self.final_dielectric_tensor = out



    def calculate_final_dielectric_tensor_22052026(self) -> None:
        """
        Return the spatial-dependent dielectric tensor including XMCD and XMLD components
        multiplies the correct elements of the dielectric tensors with the correct components of the magnetization,
        to produce the magnetization dependent dielctric tensor.

        Some efficient safety checks save time by doing these operations only when necessary

        Returns
        -------
        ndarray of shape (Nz,Ny,Nx, 2, 2)
            Thickness-weighted mean dielectric tensor.
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
        """
        eps = 1e-12
        reals = [n.real for n in self.layer_refractive_indices]
        imags = [n.imag for n in self.layer_refractive_indices]
        norm_real = mcolors.Normalize(vmin=min(reals), vmax=max(reals) + eps)
        norm_imag = mcolors.Normalize(vmin=min(imags), vmax=max(imags) + eps)
        cmap = plt.cm.viridis_r

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        fig.suptitle(f"Structure: {self.name}")
        ax[0].set_title("Real (Refraction)")
        ax[1].set_title("Imaginary (Absorption)")

        y_bottom = 0
        for name, thickness, refractive_index in zip(
            self.layer_names, self.layer_thicknesses, self.layer_refractive_indices
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
        self.shape = shape
        self.aperture_design = np.ones(shape)
        self.pixel_size = real_space_pixel_size

        self.calc_real_space_coordinates()
        self.extent_real = self.get_illumination_extent_real_space()

    def calc_real_space_coordinates(self) -> None:
        """Compute real-space (x, y) coordinate grids for the aperture.

        Sets ``self.x`` and ``self.y`` as 2-D arrays of physical
        coordinates in metres, centred on the optical axis.
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
            Standard deviation of the Gaussian smoothing filter in pixels.
            No smoothing when ``None`` or ``0``.
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
        """
        return self.aperture_design

    def visualize_aperture(self) -> None:
        """Display the current aperture design as an image."""
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
            Standard deviation of the Gaussian smoothing filter in pixels.
            No smoothing when ``None`` or ``0``.
        top_radius_factor : float, optional
            Ratio between the aperture radius at the top surface and the base
            radius at ``depth``. ``1`` gives the old cylindrical aperture;
            ``2`` gives a cone whose top opening is twice the base radius.
        taper_depth : float or None, optional
            Physical depth over which the aperture tapers from the top radius
            to the base radius. If ``None``, the taper spans the full drilled
            depth, matching the original conical behaviour.
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
        """Return a tight y/x bounding box for an aperture hole."""
        _, ny, nx = shape
        ellipticity = float(ellipticity)
        if ellipticity <= 0:
            raise ValueError(f"ellipticity must be positive, got {ellipticity}")

        radius_y = radius * np.sqrt(ellipticity)
        radius_x = radius / np.sqrt(ellipticity)
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
        """Create a possibly elliptical and rough aperture-hole mask."""
        _, ny, nx = shape
        dy = np.arange(ny, dtype=float)[:, None] - center[0]
        dx = np.arange(nx, dtype=float)[None, :] - center[1]

        ellipticity = float(ellipticity)
        if ellipticity <= 0:
            raise ValueError(f"ellipticity must be positive, got {ellipticity}")

        radius_y = radius * np.sqrt(ellipticity)
        radius_x = radius / np.sqrt(ellipticity)
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        xr = cos_angle * dx + sin_angle * dy
        yr = -sin_angle * dx + cos_angle * dy
        normalized_radius = np.sqrt((xr / radius_x) ** 2 + (yr / radius_y) ** 2)

        if roughness > 0:
            polar_angle = np.arctan2(yr / radius_y, xr / radius_x)
            boundary = np.ones((ny, nx), dtype=float)
            rng = np.random.default_rng(seed)
            min_mode, max_mode = roughness_modes
            for mode in range(max(1, int(min_mode)), int(max_mode) + 1):
                amplitude = rng.normal(scale=roughness / mode)
                phase = rng.uniform(0.0, 2.0 * np.pi)
                boundary += amplitude * np.cos(mode * polar_angle + phase)
            boundary = np.clip(
                boundary, 1.0 - 2.0 * roughness, 1.0 + 2.0 * roughness
            )
            mask = (normalized_radius <= boundary).astype(float)
        else:
            mask = (normalized_radius <= 1.0).astype(float)

        if sigma is not None and sigma != 0:
            mask = gaussian_filter(mask, sigma)
        return mask

    def create_empty_aperture(self) -> None:
        """Create an empty aperture mask (all zeros) and store it in ``self.aperture_design``."""
        self.aperture_design = np.zeros(self.shape)

    def return_aperture_mask(self) -> NDArray[np.float64]:
        """Return the current aperture design as a NumPy array.

        Returns
        -------
        NDArray[np.float64]
            2-D array of shape ``self.shape`` with values in [0, 1] representing
            the aperture mask.
        """
        return self.aperture_design

    def visualize_aperture(self) -> None:
        """Display the current aperture design as an image."""
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
        """
        Parameters
        ----------
        magnetic_structure : Structure
            Fully assembled :class:`Structure` whose
            ``effective_refractive_index`` carries the magnetic contribution.
        magnetic_pattern : ndarray
            2-D (or 3-D) magnetisation pattern with values in ``[-1, 1]``.
            A 3-D array is interpreted as ``(depth, rows, cols)``.
        """
        self.magnetic_structure = magnetic_structure
        self.magnetic_refractive_index = magnetic_structure.effective_refractive_index
        self.magnetic_pattern = magnetic_pattern

    def calc_projection_approximation(self) -> None:
        """Compute the projection approximation of the magnetisation pattern.

        For a 3-D pattern ``(depth, rows, cols)``, sums along the depth axis
        to obtain a 2-D projected map. For a 2-D pattern the input is used
        directly. The result is stored in ``self.magnetic_projection``.
        """
        if self.magnetic_pattern.ndim > 2:
            self.magnetic_projection = np.sum(self.magnetic_pattern, axis=0)
        else:
            self.magnetic_projection = self.magnetic_pattern

    def visualize_magnetic_projection(self) -> None:
        """Visualize the magnetic projection approximation."""
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
        """Visualize the magnetic dichroism and birefringence contributions."""
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

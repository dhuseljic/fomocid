from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from scattering_calculator.utils.masking import circle_mask


from pathlib import Path
from scipy.interpolate import interp1d

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple


# ============================================================
# Data structures
# ============================================================


@dataclass(frozen=True)
class Layer:
    material: str
    thickness: float

    # @property
    # def thickness(self) -> float:
    #    return self.thickness_nm * 1e-9

    def to_txt_line(self) -> str:
        return f"{self.material} {format_nm_as_meter_string(self.thickness_)}"


@dataclass
class MultilayerRecipe:
    recipe_string: str
    layers: List[Layer]
    sample_name: str | None = None
    comments: List[str] = field(default_factory=list)

    # @property
    # def total_thickness_nm(self) -> float:
    #    return sum(layer.thickness_nm for layer in self.layers)

    @property
    def total_thickness(self) -> float:
        return sum(layer.thickness for layer in self.layers)

    def add_comment(self, text: str) -> None:
        self.comments.append(text)

    def summary(self) -> str:
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

    def to_txt(
        self,
        include_header: bool = False,
        include_metadata: bool = True,
    ) -> str:
        """
        Return expanded multilayer stack as plain text.

        If include_header is False:
            Pt 5e-9
            Pt 2e-9
            Co 1e-9
            ...

        If include_header is True:
            # Sample: ...
            # Recipe: ...
            # Total thickness (nm): ...
            # Comment: ...
            Pt 5e-9
            ...
        """
        lines: List[str] = []

        if include_header and include_metadata:
            if self.sample_name:
                lines.append(f"# Sample: {self.sample_name}")
            lines.append(f"# Recipe: {self.recipe_string}")
            # lines.append(f"# Total thickness (nm): {self.total_thickness_nm:.6g}")
            lines.append(f"# Total thickness (m): {self.total_thickness:.6g}")
            for comment in self.comments:
                lines.append(f"# Comment: {comment}")

        lines.extend(layer.to_txt_line() for layer in self.layers)
        return "\n".join(lines) + "\n"

    def write_txt(
        self,
        filename: str | Path,
        include_header: bool = False,
        include_metadata: bool = True,
    ) -> Path:
        path = Path(filename)
        path.write_text(
            self.to_txt(
                include_header=include_header,
                include_metadata=include_metadata,
            ),
            encoding="utf-8",
        )
        return path


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
        Pt(5)/[Pt(2)/Co(1)]x10/Ta(5)
        [Pt(2)/[Co(1)/Ni(0.5)]x3]x5

    Conventions:
    - Thickness is assumed to be in nm
    - Layers are separated by '/'
    - Repeated blocks use [ ... ]xN
    - Nested repeated blocks are supported
    """

    def __init__(self, text: str):
        self.text = text.replace(" ", "")
        self.pos = 0

    def parse(self) -> List[Layer]:
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

            layer = self._parse_layer()
            layers.append(layer)

        return layers

    def _parse_block(self) -> List[Layer]:
        self._expect("[")
        inner_layers = self._parse_sequence(stop_char="]")
        self._expect("]")

        self._expect("x")
        repeat = self._parse_integer()

        return inner_layers * repeat

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
    parser = RecipeParser(recipe)
    layers = parser.parse()

    return MultilayerRecipe(
        recipe_string=recipe,
        layers=layers,
        sample_name=sample_name,
        comments=comments[:] if comments else [],
    )


def recipe_to_txt_file(
    recipe: str,
    filename: str | Path,
    sample_name: str | None = None,
    comments: List[str] | None = None,
    include_header: bool = False,
) -> Path:
    multilayer = parse_recipe(
        recipe=recipe,
        sample_name=sample_name,
        comments=comments,
    )
    return multilayer.write_txt(
        filename=filename,
        include_header=include_header,
        include_metadata=True,
    )


# ============================================================
# CLI
# ============================================================


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Parse and export multilayer recipes to expanded txt files."
    )

    parser.add_argument(
        "recipe",
        type=str,
        help='Recipe string, e.g. "Pt(5)/[Pt(2)/Co(1)]x10/Ta(5)"',
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="multilayer_recipe.txt",
        help="Output txt filename",
    )
    parser.add_argument(
        "--sample",
        type=str,
        default=None,
        help="Optional sample name",
    )
    parser.add_argument(
        "--comment",
        action="append",
        default=[],
        help="Optional comment. Can be passed multiple times.",
    )
    parser.add_argument(
        "--header",
        action="store_true",
        help="Include metadata header in txt output",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print recipe summary to terminal",
    )

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    multilayer = parse_recipe(
        recipe=args.recipe,
        sample_name=args.sample,
        comments=args.comment,
    )

    multilayer.write_txt(
        filename=args.output,
        include_header=args.header,
        include_metadata=True,
    )

    if args.summary:
        print(multilayer.summary())

    print(f"Wrote txt file: {args.output}")


if __name__ == "__main__":
    main()


class material_params:
    """Database of complex refractive indices for a set of materials.

    Can be initialized either with a pre-computed dictionary of refractive indices,
    or with material names and x-ray energy to load indices from the database.

    Parameters
    ----------
    refractive_indices : dict[str, complex], optional
        Mapping from element/material name to its complex refractive index.
    materials : list[str], optional
        List of material names to load from database.
    x_ray_energy : float, optional
        X-ray energy in eV for database lookups.

    Attributes
    ----------
    elements : KeysView[str]
        Names of all materials in the database.
    database : dict[str, complex]
        Full refractive index lookup table.
    x_ray_energy : float or None
        The x-ray energy used to load indices from database.
    """

    def __init__(
        self,
        refractive_indices: dict[str, complex] = None,
        materials: list[str] = None,
        x_ray_energy: float = None,
    ) -> None:

        # If both refractive_indices and materials are provided, use refractive_indices (backward compatible)
        if refractive_indices is not None:
            self.database = refractive_indices
            self.x_ray_energy = None
        elif materials is not None and x_ray_energy is not None:
            # Load from database using material names and energy
            self.x_ray_energy = x_ray_energy
            self.database = self._load_refractive_indices_from_db(
                materials, x_ray_energy
            )
        else:
            raise ValueError(
                "Either provide 'refractive_indices' dict, or both 'materials' list and 'x_ray_energy'"
            )

        self.elements = self.database.keys()

    @staticmethod
    def load_refractive_index(material_name, energy, db_path=None):
        """
        Load refractive index from database for a given material and energy.
        Uses interpolation if energy falls between database values.

        Parameters:
        -----------
        material_name : str
            Name of the material (e.g., 'Co', 'Ta', 'SiN')
        energy : float
            X-ray energy in eV
        db_path : str or Path, optional
            Path to the database directory. If None, looks for it in the project.

        Returns:
        --------
        complex
            Refractive index n = 1 - delta - i*beta
        """
        if db_path is None:
            # Try to find database relative to current working directory
            db_path = Path(
                "../src/scattering_calculator/database/material_parameter/refractive_indexes"
            )
        else:
            db_path = Path(db_path)

        # Special cases
        if material_name == "vacuum":
            return (1.0 + 0j, 0j, 0j)
        elif material_name == "perfect_absorption_mask":
            return (-1j * 1e6, 0j, 0j)

        # Map material names to folder names
        material_mapping = {
            "SiN": "Si3N4",
            "Si3N4": "Si3N4",
            "Co": "Co",
            "Ta": "Ta",
            "Pt": "Pt",
            "Au": "Au",
            "Fe": "Fe",
            "Ni": "Ni",
            "Cr": "Cr",
            "Cu": "Cu",
            "Ir": "Ir",
            "MgO": "MgO",
        }

        folder_name = material_mapping.get(material_name, material_name)
        material_dir = db_path / folder_name

        # if not txt_file.exists():
        #    # Try to find any .txt file in the directory
        #    txt_files = list(material_dir.glob("*.txt"))
        #    if txt_files:
        #        txt_file = txt_files[0]
        #    else:
        #        raise FileNotFoundError(f"No refractive index data found for {material_name} in {material_dir}")

        if material_name == "Co" and energy > 770 and energy < 805:
            txt_file = material_dir / f"{folder_name}_delta_c.txt"
            data = np.loadtxt(txt_file, skiprows=2)
            energies = data[:, 0]
            delta_c = data[:, 1] * 1e-3  # real part
            interp_delta_c = interp1d(
                energies, delta_c, kind="linear", fill_value="extrapolate"
            )

            txt_file = material_dir / f"{folder_name}_beta_c.txt"
            data = np.loadtxt(txt_file, skiprows=2)
            energies = data[:, 0]
            beta_c = data[:, 1] * 1e-3  # imaginary part
            interp_beta_c = interp1d(
                energies, beta_c, kind="linear", fill_value="extrapolate"
            )

            txt_file = material_dir / f"{folder_name}_delta.txt"
            data = np.loadtxt(txt_file, skiprows=2)
            energies = data[:, 0]
            delta = data[:, 1] * 1e-3  # real part
            interp_delta = interp1d(
                energies, delta, kind="linear", fill_value="extrapolate"
            )

            txt_file = material_dir / f"{folder_name}_beta.txt"
            data = np.loadtxt(txt_file, skiprows=2)
            energies = data[:, 0]
            beta = data[:, 1] * 1e-3  # imaginary part
            interp_beta = interp1d(
                energies, beta, kind="linear", fill_value="extrapolate"
            )

        else:
            # Find the txt file - look for one with just the material name
            txt_file = material_dir / f"{folder_name}.txt"

            # Load the data, skipping header lines
            data = np.loadtxt(txt_file, skiprows=2)
            energies = data[:, 0]
            delta = data[:, 1]  # real part
            beta = data[:, 2]  # imaginary part
            # Create interpolation functions
            interp_delta = interp1d(
                energies, delta, kind="linear", fill_value="extrapolate"
            )
            interp_beta = interp1d(
                energies, beta, kind="linear", fill_value="extrapolate"
            )
            interp_delta_c = interp1d(
                energies, delta * 0, kind="linear", fill_value="extrapolate"
            )
            interp_beta_c = interp1d(
                energies, beta * 0, kind="linear", fill_value="extrapolate"
            )

        # interp_delta_l = interp1d(energies, delta*0, kind='linear', fill_value='extrapolate')
        # interp_beta_l = interp1d(energies, beta*0, kind='linear', fill_value='extrapolate')

        # Get values at the requested energy
        delta_at_energy = float(interp_delta(energy))
        beta_at_energy = float(interp_beta(energy))
        delta_c_at_energy = float(interp_delta_c(energy))
        beta_c_at_energy = float(interp_beta_c(energy))
        delta_l_at_energy = 0  # float(interp_delta_l(energy))
        beta_l_at_energy = 0  # float(interp_beta_l(energy))

        # Return complex refractive index
        # n = 1 - delta - i*beta (following X-ray optics convention)
        return np.array(
            [
                1.0 - delta_at_energy - 1j * beta_at_energy,
                -delta_c_at_energy - 1j * beta_c_at_energy,
                -delta_l_at_energy - 1j * beta_l_at_energy,
            ]
        )

    def _load_refractive_indices_from_db(
        self, materials: list[str], x_ray_energy: float, db_path=None
    ) -> dict:
        """
        Load refractive indices for multiple materials from the database.

        Parameters
        ----------
        materials : list[str]
            List of material names to load.
        x_ray_energy : float
            X-ray energy in eV.
        db_path : str or Path, optional
            Path to the database directory.

        Returns
        -------
        dict
            Dictionary with material names as keys and refractive indices as values.
        """
        refractive_indices = {}
        for material in materials:
            try:
                result = self.load_refractive_index(
                    material, x_ray_energy, db_path=db_path
                )
                # Handle both single and tuple returns
                # if isinstance(result, tuple):
                #    refractive_indices[material] = result[0]
                # else:
                refractive_indices[material] = result
            except FileNotFoundError as e:
                print(f"Warning: {e}")

        return refractive_indices

    def get_refractive_indices_dict(self, x_ray_energy, materials=None, db_path=None):
        """
        Get refractive indices for all materials at a specific X-ray energy.

        Parameters:
        -----------
        x_ray_energy : float
            X-ray energy in eV
        materials : list, optional
            List of material names. If None, uses default set.
        db_path : str or Path, optional
            Path to the database directory.

        Returns:
        --------
        dict
            Dictionary with material names as keys and complex refractive indices as values
        """
        if materials is None:
            materials = ["vacuum", "perfect_absorption_mask", "SiN", "Ta", "Co"]

        refractive_indices = {}
        for material in materials:
            try:
                result = self.load_refractive_index(
                    material, x_ray_energy, db_path=db_path
                )
                # Handle both single and tuple returns
                if isinstance(result, tuple):
                    refractive_indices[material] = result[0]
                else:
                    refractive_indices[material] = result
            except FileNotFoundError as e:
                print(f"Warning: {e}")

        return refractive_indices

    def get_refractive_index(
        self, elements: str | list[str]
    ) -> complex | list[complex]:
        """Look up the complex refractive index for one or more materials.

        Parameters
        ----------
        elements : str or list of str
            Single material name or list of material names.

        Returns
        -------
        complex or list of complex
            Single refractive index when ``elements`` is a string;
            list of refractive indices when ``elements`` is a list.
        """
        if isinstance(elements, str):
            return self.database[elements]
        return [self.database[element] for element in elements]


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
    layer_names : list of str
        Element name of each layer in deposition order.
    layer_thicknesses : list of float
        Physical thickness of each layer in metres.
    layer_refractive_indices : list of complex
        Complex refractive index of each layer.
    effective_refractive_indices : list of complex
        Effective index (``n * thickness``) of each layer.
    """

    def __init__(self, name: str, material_params: material_params) -> None:
        self.name = name
        self.material_params = material_params

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

    def dielectric_tensor_mixed(self, n, theta=0.0):
        """
        Build a 2x2 transverse dielectric tensor from:
        n=(n0,dn_l,dn_c)
        - n0: baseline isotropic refractive index
        - dn_lin: linear anisotropy contribution
        - dn_circ: circular anisotropy contribution
        - theta: rotation angle (radians) of the linear principal axes

        Small-anisotropy approximation:
            eps0      = n0^2
            eps_l   = 2 n0 dn_l
            eps_c  = 2 n0 dn_c
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

    def create_2d_refractive_index_map(self, shape: tuple) -> np.ndarray:
        """Create a 2D array representing the refractive index profile of the structure.

        Parameters
        ----------
        shape : tuple of int
            Desired shape of the output array (height, width).

        Returns
        -------
        np.ndarray
            2D array of shape `shape` where each row corresponds to a layer's
            refractive index, repeated across the width.
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
        Thickness-weighted effective transverse dielectric tensor.

        Parameters
        ----------
        layer_tensors : list of (2,2) complex arrays
            Dielectric tensors of the layers.
        layer_thicknesses : list of float
            Thicknesses of the layers.

        Returns
        -------
        eps_eff : (2,2) complex array
            Effective dielectric tensor.
        """

        D = np.sum(self.layer_thicknesses)
        eps_eff = np.zeros_like(np.asarray(self.dielectric_tensors[0], dtype=complex))

        for eps, d in zip(self.dielectric_tensors, self.layer_thicknesses):
            eps_eff += d * np.asarray(eps, dtype=complex)
        eps_eff /= D

        return eps_eff

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


class Apertures:
    """Class for generating masks from circular apertures.

    Parameters
    ----------
    None

    Attributes
    ----------
    None
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

        x = (np.arange(self.shape[1]) - self.shape[1] / 2) * self.pixel_size
        y = (np.arange(self.shape[0]) - self.shape[0] / 2) * self.pixel_size
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
        self.magnetic_structure = magnetic_structure
        self.magnetic_refractive_index = magnetic_structure.effective_refractive_index
        self.magnetic_pattern = magnetic_pattern

    def calc_projection_approximation(self) -> ArrayLike:
        """Calculate the projection approximation for a given magnetic refractive index.

        Parameters
        ----------
        magnetic_refractive_index : complex
            Complex magnetization profile for a layer.

        Returns
        -------
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

    def calc_magnetic_dichroism_birefringence(self):
        """Calculate the magnetic dichroism and birefringence contributions to the refractive index."""
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

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from pathlib import Path
from scipy.interpolate import interp1d


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
    def load_refractive_index(
        material_name: str,
        energy: float,
        db_path: str | Path | None = None,
    ) -> NDArray[np.complex128]:
        """Load refractive index from the database for a given material and energy.

        Uses linear interpolation when the requested energy falls between
        database values. For Co in the L-edge range (770–805 eV) the
        circular dichroism files are loaded in addition to the total index.

        Parameters
        ----------
        material_name : str
            Name of the material, e.g. ``"Co"``, ``"Ta"``, ``"SiN"``.
            ``"vacuum"`` and ``"perfect_absorption_mask"`` are handled as
            special cases.
        energy : float
            X-ray energy in eV.
        db_path : str or Path or None, optional
            Path to the refractive-index database directory. Defaults to
            ``../src/scattering_calculator/database/material_parameter/refractive_indexes``
            relative to the current working directory.

        Returns
        -------
        ndarray of shape (3,) and dtype complex128
            ``[n_total, n_circ, n_lin]`` where
            ``n_total = 1 - delta - i*beta`` (X-ray optics convention),
            ``n_circ = -delta_c - i*beta_c``, and ``n_lin = 0`` (reserved).
        """
        if db_path is None:
            db_path = Path(
                "../src/scattering_calculator/database/material_parameter/refractive_indexes"
            )
        else:
            db_path = Path(db_path)

        if material_name == "vacuum":
            return np.array([1.0 + 0j, 0j, 0j])
        elif material_name == "perfect_absorption_mask":
            return np.array([-1j * 1e6, 0j, 0j])

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

        if material_name == "Co" and 770 < energy < 805:
            data = np.loadtxt(material_dir / f"{folder_name}_delta_c.txt", skiprows=2)
            interp_delta_c = interp1d(data[:, 0], data[:, 1] * 1e-3, kind="linear", fill_value="extrapolate")

            data = np.loadtxt(material_dir / f"{folder_name}_beta_c.txt", skiprows=2)
            interp_beta_c = interp1d(data[:, 0], data[:, 1] * 1e-3, kind="linear", fill_value="extrapolate")

            data = np.loadtxt(material_dir / f"{folder_name}_delta.txt", skiprows=2)
            interp_delta = interp1d(data[:, 0], data[:, 1] * 1e-3, kind="linear", fill_value="extrapolate")

            data = np.loadtxt(material_dir / f"{folder_name}_beta.txt", skiprows=2)
            interp_beta = interp1d(data[:, 0], data[:, 1] * 1e-3, kind="linear", fill_value="extrapolate")
        else:
            data = np.loadtxt(material_dir / f"{folder_name}.txt", skiprows=2)
            interp_delta = interp1d(data[:, 0], data[:, 1], kind="linear", fill_value="extrapolate")
            interp_beta = interp1d(data[:, 0], data[:, 2], kind="linear", fill_value="extrapolate")
            interp_delta_c = interp1d(data[:, 0], data[:, 1] * 0, kind="linear", fill_value="extrapolate")
            interp_beta_c = interp1d(data[:, 0], data[:, 2] * 0, kind="linear", fill_value="extrapolate")

        return np.array(
            [
                1.0 - float(interp_delta(energy)) - 1j * float(interp_beta(energy)),
                -float(interp_delta_c(energy)) - 1j * float(interp_beta_c(energy)),
                0j,
            ]
        )

    def _load_refractive_indices_from_db(
        self, materials: list[str], x_ray_energy: float, db_path: str | Path | None = None
    ) -> dict:
        """Load refractive indices for multiple materials from the database.

        Parameters
        ----------
        materials : list[str]
            List of material names to load.
        x_ray_energy : float
            X-ray energy in eV.
        db_path : str or Path or None, optional
            Path to the database directory.

        Returns
        -------
        dict
            Mapping from material name to refractive index array of shape (3,).
        """
        refractive_indices = {}
        for material in materials:
            try:
                refractive_indices[material] = self.load_refractive_index(
                    material, x_ray_energy, db_path=db_path
                )
            except FileNotFoundError as e:
                print(f"Warning: {e}")
        return refractive_indices

    def get_refractive_indices_dict(
        self,
        x_ray_energy: float,
        materials: list[str] | None = None,
        db_path: str | Path | None = None,
    ) -> dict:
        """Load refractive indices for a set of materials at a given energy.

        Parameters
        ----------
        x_ray_energy : float
            X-ray energy in eV.
        materials : list of str or None, optional
            Material names to load. Defaults to
            ``["vacuum", "perfect_absorption_mask", "SiN", "Ta", "Co"]``.
        db_path : str or Path or None, optional
            Path to the database directory.

        Returns
        -------
        dict
            Mapping from material name to refractive index array of shape (3,).
        """
        if materials is None:
            materials = ["vacuum", "perfect_absorption_mask", "SiN", "Ta", "Co"]

        refractive_indices = {}
        for material in materials:
            try:
                refractive_indices[material] = self.load_refractive_index(
                    material, x_ray_energy, db_path=db_path
                )
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

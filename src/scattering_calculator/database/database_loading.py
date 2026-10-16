"""Material-parameter database loading helpers."""

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
        db_path: str | Path | None = None,
    ) -> None:

        # If both refractive_indices and materials are provided, use refractive_indices (backward compatible)
        """Initialize a material_params instance.

        Parameters
        ----------
        refractive_indices : dict[str, complex]
            Input value for ``refractive_indices``.
        materials : list[str]
            Input value for ``materials``.
        x_ray_energy : float
            Input value for ``x_ray_energy``.
        db_path : str or Path or None
            Refractive-index dataset root; None uses bundled material files.

        Returns
        -------
        None
            The function completes in place.
        """
        self.db_path = Path(db_path) if db_path is not None else None
        self.optical_response_database = {}
        if refractive_indices is not None:
            self.database = refractive_indices
            self.x_ray_energy = None
        elif materials is not None and x_ray_energy is not None:
            # Load from database using material names and energy
            self.x_ray_energy = x_ray_energy
            self.database = self._load_refractive_indices_from_db(
                materials, x_ray_energy, db_path=self.db_path
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
            the package's bundled
            ``database/material_parameter/refractive_indexes`` directory.

        Returns
        -------
        ndarray of shape (3,) and dtype complex128
            ``[n_total, n_circ, n_lin]`` where
            ``n_total = 1 - delta - i*beta`` (X-ray optics convention),
            ``n_circ = -delta_c - i*beta_c``. Named magnetic circular/linear
            channel tables override historical values when present; otherwise
            the original Co circular response and zero linear channel remain.
        """
        if db_path is None:
            db_path = (
                Path(__file__).resolve().parent
                / "material_parameter"
                / "refractive_indexes"
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

        legacy_co_files = ("delta_c", "beta_c", "delta", "beta")
        if (
            material_name == "Co"
            and 770 < energy < 805
            and all(
                (material_dir / f"{folder_name}_{suffix}.txt").is_file()
                for suffix in legacy_co_files
            )
        ):
            data = np.loadtxt(material_dir / f"{folder_name}_delta_c.txt", skiprows=2)
            interp_delta_c = interp1d(
                data[:, 0], data[:, 1] * 1e-3, kind="linear", fill_value="extrapolate"
            )

            data = np.loadtxt(material_dir / f"{folder_name}_beta_c.txt", skiprows=2)
            interp_beta_c = interp1d(
                data[:, 0], data[:, 1] * 1e-3, kind="linear", fill_value="extrapolate"
            )

            data = np.loadtxt(material_dir / f"{folder_name}_delta.txt", skiprows=2)
            interp_delta = interp1d(
                data[:, 0], data[:, 1] * 1e-3, kind="linear", fill_value="extrapolate"
            )

            data = np.loadtxt(material_dir / f"{folder_name}_beta.txt", skiprows=2)
            interp_beta = interp1d(
                data[:, 0], data[:, 1] * 1e-3, kind="linear", fill_value="extrapolate"
            )
        else:
            data = np.loadtxt(material_dir / f"{folder_name}.txt", skiprows=2)
            interp_delta = interp1d(
                data[:, 0], data[:, 1], kind="linear", fill_value="extrapolate"
            )
            interp_beta = interp1d(
                data[:, 0], data[:, 2], kind="linear", fill_value="extrapolate"
            )
            interp_delta_c = interp1d(
                data[:, 0], data[:, 1] * 0, kind="linear", fill_value="extrapolate"
            )
            interp_beta_c = interp1d(
                data[:, 0], data[:, 2] * 0, kind="linear", fill_value="extrapolate"
            )

        # Named material channels use the same energy lookup as the historical
        # magnetic files. Missing optional magnetic channels retain legacy values.
        circular = material_params.load_optical_response(
            material_name, "magnetic_circular", energy, db_path, required=False
        )
        linear = material_params.load_optical_response(
            material_name, "magnetic_linear", energy, db_path, required=False
        )
        return np.array(
            [
                1.0 - float(interp_delta(energy)) - 1j * float(interp_beta(energy)),
                (-float(interp_delta_c(energy)) - 1j * float(interp_beta_c(energy)))
                if circular is None
                else circular,
                0j if linear is None else linear,
            ]
        )

    @staticmethod
    def optical_response_path(material, channel, db_path=None):
        """Locate a named channel beside the material's refractive-index files.

        Parameters
        ----------
        material : str
            Material dataset name; SiN aliases Si3N4 as in the base loader.
        channel : str
            Magnetic, natural or ferroelectric response channel.
        db_path : str or Path or None
            Dataset root, or None for the bundled refractive-index datasets.

        Returns
        -------
        Path
            Three-column energy/delta/beta response table.
        """
        channels = (
            "magnetic_circular",
            "magnetic_linear",
            "natural_linear",
            "natural_circular",
            "ferroelectric",
        )
        if channel not in channels:
            raise ValueError(f"Unknown material response channel: {channel!r}")
        root = (
            Path(db_path)
            if db_path is not None
            else (
                Path(__file__).resolve().parent
                / "material_parameter"
                / "refractive_indexes"
            )
        )
        name = "Si3N4" if material == "SiN" else material
        if Path(name).name != name or name in (".", ".."):
            raise ValueError("Material must be a dataset name, not a path")
        return root / name / f"{name}_{channel}.txt"

    @staticmethod
    def load_optical_response(
        material, channel, energy, db_path=None, *, required=True
    ):
        """Interpolate one dataset channel as -delta(E)-i beta(E).

        Parameters
        ----------
        material, channel : str
            Material name and response channel.
        energy : float
            Physical photon energy in eV.
        db_path : str or Path or None
            Same dataset root used for isotropic and magnetic optical constants.
        required : bool
            Raise on missing data; False is for legacy optional magnetic channels.

        Returns
        -------
        complex or None
            Dimensionless index increment. No channel extrapolation is allowed.
        """
        path = material_params.optical_response_path(material, channel, db_path)
        if not path.is_file():
            if not required:
                return None
            raise FileNotFoundError(
                f"Missing {channel} refractive-index dataset for {material}: {path}"
            )
        table = np.loadtxt(path, skiprows=2, ndmin=2)
        if (
            table.ndim != 2
            or table.shape[0] < 2
            or table.shape[1] != 3
            or not np.isfinite(table).all()
            or np.any(table[:, 0] <= 0)
            or np.any(np.diff(table[:, 0]) <= 0)
        ):
            raise ValueError(
                f"Expected increasing positive energy_eV, delta, beta columns in {path}"
            )
        if (
            energy is None
            or not np.isfinite(energy)
            or not table[0, 0] <= energy <= table[-1, 0]
        ):
            raise ValueError(
                f"Photon energy {energy} eV is outside material response dataset {path}"
            )
        delta = np.interp(energy, table[:, 0], table[:, 1])
        beta = np.interp(energy, table[:, 0], table[:, 2])
        return complex(-delta, -beta)

    def get_optical_response(self, material, channel):
        """Look up a response from this run's energy-dependent material database.

        Parameters
        ----------
        material, channel : str
            Material and named optical channel; spatial order is supplied separately.

        Returns
        -------
        complex
            Cached dataset value at the material database's current energy.
        """
        key = (material, channel)
        if key not in self.optical_response_database:
            self.optical_response_database[key] = self.load_optical_response(
                material, channel, self.x_ray_energy, self.db_path
            )
        return self.optical_response_database[key]

    def _load_refractive_indices_from_db(
        self,
        materials: list[str],
        x_ray_energy: float,
        db_path: str | Path | None = None,
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

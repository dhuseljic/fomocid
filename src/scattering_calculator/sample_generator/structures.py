from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from scattering_calculator.utils.masking import circle_mask


class material_params:
    """Database of complex refractive indices for a set of materials.

    Parameters
    ----------
    refractive_indices : dict[str, complex]
        Mapping from element/material name to its complex refractive index.

    Attributes
    ----------
    elements : KeysView[str]
        Names of all materials in the database.
    database : dict[str, complex]
        Full refractive index lookup table.
    """

    def __init__(self, refractive_indices: dict[str, complex]) -> None:
        self.elements = refractive_indices.keys()
        self.database = refractive_indices

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
        """
        refractive_index = self.material_params.get_refractive_index(element)
        effective_index = self.calc_effective_refractive_indices(
            refractive_index, thickness
        )

        self.layer_names.append(element)
        self.layer_thicknesses.append(thickness)
        self.layer_refractive_indices.append(refractive_index)
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
            self.effective_refractive_indices.pop(i)
        return len(indices)

    def return_total_effective_refractive_index(self) -> complex:
        """Return the sum of effective refractive indices across all layers.

        Returns
        -------
        complex
            Sum of ``n * thickness`` over all layers.
        """
        return sum(self.effective_refractive_indices)

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
        for name, thickness_m, refractive_index in zip(
            self.layer_names, self.layer_thicknesses, self.layer_refractive_indices
        ):
            thickness = thickness_m * 1e9  # convert to nm for visualization
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

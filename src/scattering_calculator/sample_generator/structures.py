import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


class material_params:
    def __init__(self, refractive_indices: dict[str, complex]) -> None:
        self.elements = refractive_indices.keys()
        self.database = refractive_indices

    def get_refractive_index(
        self, elements: str | list[str]
    ) -> complex | list[complex]:
        if isinstance(elements, str):
            return self.database[elements]
        return [self.database[element] for element in elements]


class Structure:
    def __init__(self, name: str, material_params: material_params) -> None:
        self.name = name
        self.material_params = material_params

        self.layer_names: list[str] = []
        self.layer_thicknesses: list = []
        self.layer_refractive_indices: list = []
        self.effective_refractive_indices: list = []

    def calc_effective_refractive_indices(
        self, refractive_index: complex, thickness: float
    ) -> float:
        return refractive_index * thickness

    def add_layer(self, element: str, thickness: float) -> None:
        refractive_index = self.material_params.get_refractive_index(element)
        effective_index = self.calc_effective_refractive_indices(
            refractive_index, thickness
        )

        self.layer_names.append(element)
        self.layer_thicknesses.append(thickness)
        self.layer_refractive_indices.append(refractive_index)
        self.effective_refractive_indices.append(effective_index)

    def return_layer_refractive_indices(self):
        return np.array(self.layer_refractive_indices)

    def return_total_effective_refractive_index(self) -> complex:
        return sum(self.effective_refractive_indices)

    def visualize_structure(self) -> None:
        reals = [n.real for n in self.layer_refractive_indices]
        imags = [n.imag for n in self.layer_refractive_indices]
        norm_real = mcolors.Normalize(vmin=min(reals), vmax=max(reals))
        norm_imag = mcolors.Normalize(vmin=min(imags), vmax=max(imags))
        cmap = plt.cm.viridis_r

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        fig.suptitle(f"Structure: {self.name}")
        ax[0].set_title("Real")
        ax[1].set_title("Imaginary")

        y_bottom = 0
        for i in range(len(self.layer_thicknesses)):
            thickness = (
                self.layer_thicknesses[i] * 1e9
            )  # convert to nm for visualization
            refractive_index = self.layer_refractive_indices[i]
            name = self.layer_names[i]
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

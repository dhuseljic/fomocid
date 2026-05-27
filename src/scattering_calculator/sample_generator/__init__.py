"""Sample generation for coherent scattering simulations."""

from .gray_scott_generator import (
    MORPHOLOGY_REGIONS,
    GrayScottBatch,
    GrayScottConfig,
    generate,
    sample_fk,
)
from .pattern_generator import (
    create_lattice,
    create_binary_labyrinth_pattern,
    create_disordered_skyrmion_lattice_pattern,
    create_image_pattern,
    create_saturated_pattern,
    create_skyrmion_pattern,
    create_stripe_pattern,
    create_wavy_stripe_pattern,
    map_magnetization_to_3d,
    skyrmions_on_lattice,
)
from .structures import (
    Apertures2D,
    Apertures3D,
    Layer,
    Magnetic_Structure,
    MultilayerRecipe,
    RecipeParser,
    Structure,
    parse_recipe,
)
from scattering_calculator.database.database_loading import material_params

__all__ = [
    "Apertures2D",
    "Apertures3D",
    "GrayScottBatch",
    "GrayScottConfig",
    "Layer",
    "Magnetic_Structure",
    "MORPHOLOGY_REGIONS",
    "MultilayerRecipe",
    "RecipeParser",
    "Structure",
    "create_lattice",
    "create_binary_labyrinth_pattern",
    "create_disordered_skyrmion_lattice_pattern",
    "create_image_pattern",
    "create_saturated_pattern",
    "create_skyrmion_pattern",
    "create_stripe_pattern",
    "create_wavy_stripe_pattern",
    "generate",
    "map_magnetization_to_3d",
    "material_params",
    "parse_recipe",
    "sample_fk",
    "skyrmions_on_lattice",
]

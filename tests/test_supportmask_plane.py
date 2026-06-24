"""Tests for support-mask plane selection."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scattering_calculator.simulation_pipelines.simulation_configuration import (
    FrontApertureConfig,
)


def test_supportmask_uses_gold_layer_before_silicon_nitride() -> None:
    config = FrontApertureConfig(
        aperture_method="FTH_circular",
        aperture_shape=(3, 80, 80),
        real_space_pixel_size=1e-9,
        aperture_thicknesses=[1e-9, 1e-9, 1e-9],
        aperture_layer_names=["Au", "Au", "SiN"],
        use_roi=True,
        aperture_config={
            "apertures_type": ["OH"],
            "apertures_radius": [4e-9],
            "apertures_center": [(0.0, 0.0)],
            "apertures_sigma": [0.0],
            "apertures_angle": [0.0],
            "apertures_ellipticity": [1.0],
            "apertures_roughness": [0.0],
            "apertures_roughness_modes": [(0, 0)],
            "apertures_seed": [-1],
            "apertures_top_radius_factor": [3.0],
            "aperture_taper_depth": 2e-9,
            "thickness_OH": 2e-9,
        },
    )
    config.setup()

    supportmask = config.create_supportmask()
    top_opening = 1.0 - config.aperture.aperture_design[0]
    gold_before_sin_opening = 1.0 - config.aperture.aperture_design[1]

    np.testing.assert_array_equal(supportmask, gold_before_sin_opening > 0)
    assert np.count_nonzero(supportmask) < np.count_nonzero(top_opening > 0)

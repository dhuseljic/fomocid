from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scattering_calculator.experimental_conditions.detector import beamstop, detector_layout


class BeamstopAntialiasTests(unittest.TestCase):
    def test_antialias_creates_fractional_wire_edges(self) -> None:
        layout = detector_layout(
            pixel_size=1.0,
            detector_shape=(64, 64),
            distance_sample_detector=10.0,
            detector_center=(32, 32),
        )
        stop = beamstop(layout, distance_detector_beamstop=1.0)
        stop.create_circle_beamstop(
            center=(32, 32),
            radius=1.0,
            wire_width=3.0,
            angle=np.pi / 7.0,
            antialias=4,
        )

        mask = stop.return_beamstop()
        fractional = mask[(mask > 0.0) & (mask < 1.0)]

        self.assertGreater(fractional.size, 0)
        self.assertEqual(mask.shape, layout.detector_shape)


if __name__ == "__main__":
    unittest.main()

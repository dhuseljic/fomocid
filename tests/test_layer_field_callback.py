"""Streaming layer capture must preserve propagation and sample-exit fields."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from scattering_calculator.beam_propagator.simple_propagation import scalar_wavefronts
from scattering_calculator.beam_propagator.Jones_propagator import wavefronts


class LayerFieldCallbackTests(unittest.TestCase):
    def test_scalar_capture_matches_analytic_post_material_planes(self):
        wavelength = 2e-9
        thicknesses = np.array([0.2e-9, 0.3e-9, 0.4e-9])
        indices = np.array([0.97 - 0.01j, 0.99 - 0.02j, 0.96 - 0.03j])
        stack = np.broadcast_to(indices[:, None, None], (3, 5, 6))
        captured = []
        options = dict(beam_parameters=SimpleNamespace(wavelength=wavelength, pol="CR"),
                       layer_thicknesses=thicknesses, real_space_pixel_size=1e-9,
                       E_in=np.ones((5, 6), complex), refractive_index_stack=stack)
        default = scalar_wavefronts(**options)
        traced = scalar_wavefronts(**options, layer_callback=lambda i, f: captured.append((i, f.copy())))
        self.assertEqual([i for i, _ in captured], [0, 1, 2])
        for i, field in captured:
            optical_path = np.sum(indices[:i + 1] * thicknesses[:i + 1]) + np.sum(thicknesses[:i])
            np.testing.assert_allclose(field, np.exp(-2j * np.pi / wavelength * optical_path))
        np.testing.assert_array_equal(traced.exit_wave, default.exit_wave)
        np.testing.assert_array_equal(captured[-1][1], traced.exit_wave)

    def test_callbacks_preserve_full_and_roi_diffraction(self):
        rng = np.random.default_rng(9)
        field = rng.normal(size=(8, 9)) + 1j * rng.normal(size=(8, 9))
        index = np.full((3, 8, 9), 0.99 - 0.01j)
        eps = np.zeros((3, 8, 9, 2, 2), complex)
        eps[..., 0, 0] = index**2
        eps[..., 1, 1] = index**2
        for method in ("Scalar", "Jones"):
            for roi in (False, True):
                with self.subTest(method=method, roi=roi):
                    options = dict(beam_parameters=SimpleNamespace(wavelength=2e-9, pol="CR"),
                                   layer_thicknesses=[1e-9] * 3, real_space_pixel_size=1e-9,
                                   propagate=True, propagation_padding_px=2,
                                   propagation_absorber_width_px=1, propagation_absorber_strength=2,
                                   multislice_propagation_roi=roi,
                                   aperture_support_regions=((slice(1, 7), slice(1, 8)),))
                    if method == "Scalar":
                        factory = scalar_wavefronts
                        options.update(E_in=field, refractive_index_stack=index)
                    else:
                        factory = wavefronts
                        options.update(E_in=np.stack([field, 1j * field], axis=-1), eps_stack=eps,
                                       calculate_farfield=False, store_intermediate_wavefields=True)
                    captured = []
                    default = factory(**options)
                    traced = factory(**options, layer_callback=lambda i, f: captured.append(f.copy()))
                    self.assertEqual(len(captured), 3)
                    np.testing.assert_array_equal(traced.exit_wave, default.exit_wave)
                    np.testing.assert_array_equal(captured[-1], traced.exit_wave)
                    if method == "Jones":
                        np.testing.assert_array_equal(np.stack(captured), traced.intermediate_wavefields)


if __name__ == "__main__":
    unittest.main()

"""Notebook configuration and independent 3D Fourier geometry checks.

Run: python -m unittest discover -s paper/scattering_calculator -p 'test_*.py'
"""
import contextlib
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import experiments as ex


class ExperimentConfigurationTests(unittest.TestCase):
    def test_explicit_thickness_is_preserved_and_auto_design_changes(self):
        cfg = ex.SkyrmionConfig(thickness_nm=75., lattice_nm=16., radius_nm=4.)
        self.assertEqual(cfg.geometry()['thickness_nm'], 75.)
        self.assertEqual(replace(cfg, energy_eV=790.).geometry()['thickness_nm'], 75.)
        self.assertNotEqual(replace(cfg, thickness_nm=None).geometry()['thickness_nm'],
                            ex.SkyrmionConfig().geometry()['thickness_nm'])
        self.assertGreater(cfg.geometry()['normal_thickness_factor'], 0)

    def test_detector_parameters_change_ewald_coordinates(self):
        cfg = ex.SkyrmionConfig(detector_n=31)
        q, _ = ex.configured_detector(cfg)
        wide, _ = ex.configured_detector(replace(cfg, detector_pitch_m=2*cfg.detector_pitch_m))
        far, _ = ex.configured_detector(replace(cfg, detector_distance_m=2*cfg.detector_distance_m))
        self.assertGreater(abs(wide[0,0,0]), abs(q[0,0,0]))
        self.assertLess(abs(far[0,0,0]), abs(q[0,0,0]))
        shifted, _ = ex.configured_detector(replace(cfg, detector_center_offset_xy_px=(2.,-1.)))
        np.testing.assert_allclose(shifted[14,17], 0., atol=1e-15)
        k = 2*np.pi*cfg.energy_eV/ex.HC
        np.testing.assert_allclose(np.linalg.norm(shifted+[0,0,k],axis=-1), k, rtol=1e-13)

    def test_texture_radius_and_spacing_are_used(self):
        x,y=np.meshgrid(np.linspace(-15,15,64),np.linspace(-15,15,64))
        base=ex.texture(x,y,12,3.4)
        radius=ex.texture(x,y,12,4.5)
        spacing=ex.texture(x,y,16,3.4)
        self.assertGreater(np.linalg.norm(base-radius),1)
        self.assertGreater(np.linalg.norm(base-spacing),1)
        np.testing.assert_allclose(np.linalg.norm(radius,axis=-1),1,atol=1e-14)

    def test_invalid_configurations_fail_early(self):
        for kwargs in (dict(radius_nm=6.),dict(thickness_nm=-1.),dict(detector_distance_m=0.),
                       dict(scan_angles_deg=()),dict(contrast_channel='invalid')):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ex.SkyrmionConfig(**kwargs)

    def test_multislice_uses_config_and_detector_shape(self):
        cfg=ex.SkyrmionConfig(thickness_nm=12.,radius_nm=3.,lattice_nm=14.,
                             multislice_n=48,multislice_dx_nm=1.,multislice_dz_nm=3.,
                             detector_n=25,beam_sigma_nm=8.)
        result=ex.skyrmion_multislice(3.,config=cfg)
        changed=ex.skyrmion_multislice(3.,config=replace(cfg,radius_nm=4.))
        self.assertEqual(result['intensity'].shape,(48,48))
        propagating = np.isfinite(result['intensity']) & np.isfinite(changed['intensity'])
        self.assertGreater(np.linalg.norm((result['intensity']-changed['intensity'])[propagating]),0)
        measured=ex.sample_multislice_detector(result,cfg)
        self.assertEqual(measured['intensity'].shape,(25,25))
        np.testing.assert_allclose(measured['q_lab'],ex.configured_detector(cfg)[0])

    def test_fft_voxel_thickness_and_padding_convergence(self):
        cfg=ex.SkyrmionConfig(born_n=64,born_dx_nm=1.,beam_sigma_nm=10.,thickness_nm=43.7,
                             detector_n=31,detector_pitch_m=65e-6,q_bins=31,
                             scan_angles_deg=(-8.,0.,8.),fft_nz=128,fft_dz_nm=2.,contrast_channel='mz')
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            out=Path(directory)
            coarse=ex.fft_volume_comparison(out,cfg)
            fine=ex.fft_volume_comparison(out,replace(cfg,fft_nz=256))
            self.assertAlmostEqual(fine['voxelized_thickness_nm'],43.7,places=12)
            self.assertLess(fine['born_vs_sampled_fft_relative_L2'],coarse['born_vs_sampled_fft_relative_L2'])
            with np.load(out/'fft_volume_comparison.npz') as data:
                self.assertTrue(np.isnan(data['diffraction_intensity'][data['coverage']==0]).all())
                mid=cfg.q_bins//2
                u=(np.arange(cfg.born_n)-cfg.born_n//2)*cfg.born_dx_nm
                x,y=np.meshgrid(u,u)
                image=(ex.texture(x,y,cfg.lattice_nm,cfg.radius_nm)[...,2]-1)*np.exp(-(x*x+y*y)/(2*cfg.beam_sigma_nm**2))
                dc=abs(image.sum()*cfg.born_dx_nm**2*cfg.thickness_nm)**2
                np.testing.assert_allclose(data['direct_mz_intensity'][mid,mid,mid],dc,rtol=1e-6)

    def test_custom_fth_and_detector_run(self):
        cfg=ex.FTHConfig(n=64,dx_nm=10.,object_radius_nm=50.,reference_radius_nm=15.,
                         reference_xy_nm=(200.,30.),layers_nm=(('Au',20.),('Co',4.)),
                         beam_sigma_nm=180.,domain_period_nm=60.,energies_eV=(775.,780.))
        case=ex.fth_case(config=cfg)
        self.assertEqual(case['images'].shape,(2,64,64))
        self.assertEqual(case['energy_eV'],cfg.energy_eV)
        with tempfile.TemporaryDirectory() as directory:
            effects=ex.DetectorEffectsConfig(expected_peak_counts=1234.,beamstop_radius_px=2.,noise_seed=8)
            first=ex.detector_figure(Path(directory),cfg,effects)
            second=ex.detector_figure(Path(directory),cfg,effects)
            np.testing.assert_array_equal(first,second)
            metadata=json.loads((Path(directory)/'provenance.json').read_text())
            self.assertEqual(metadata['parameters']['detector']['expected_peak_counts'],1234.)


if __name__=='__main__':
    unittest.main()

# Tutorials

The numbered notebooks are the maintained coherent-scattering learning path.
Run them from the repository root after installing the project with
`python -m pip install -e .`. Each notebook also locates `src/` when opened
from the `tutorials/` directory.

## Numbered notebooks

1. [`01_material_parameters.ipynb`](01_material_parameters.ipynb) — parse a
   multilayer recipe and inspect refractive-index channels and dielectric
   tensors.
2. [`02_magnetic_patterns.ipynb`](02_magnetic_patterns.ipynb) — create and
   inspect labyrinth, stripe, bubble/skyrmion, and saturated magnetization maps.
3. [`03_binary_domain_phase_space.ipynb`](03_binary_domain_phase_space.ipynb) —
   analyze the `k0`/`eps`/`target_mean` space, classify bubbles and stripes,
   and define morphology-rich sampling regions.
4. [`04_holography_mask.ipynb`](04_holography_mask.ipynb) — construct object
   and reference holes and inspect the three-dimensional mask.
5. [`05_beamstop.ipynb`](05_beamstop.ipynb) — configure beamstop geometry,
   wires, roughness, projection, and anti-aliasing.
6. [`06_illumination.ipynb`](06_illumination.ipynb) — configure plane-wave and
   Gaussian illumination, focus, displacement, tilt, Jones polarization, and
   partially polarized Stokes states; visualize how circular polarization
   degree controls magnetic contrast.
7. [`07_hologram_generation_and_artifacts.ipynb`](07_hologram_generation_and_artifacts.ipynb)
   — generate ideal and detected holograms, apply artifacts, and inspect FTH
   reconstructions.
8. [`08_compare_propagation_formalisms.ipynb`](08_compare_propagation_formalisms.ipynb)
   — compare Scalar, Jones, and Mueller–Stokes propagation on one sample.
9. [`09_hologram_pipeline.ipynb`](09_hologram_pipeline.ipynb) — configure and
   run `HologramPipeline`, sample parameter ranges, and read the HDF5 output.
10. [`10_multislice_and_roi_modes.ipynb`](10_multislice_and_roi_modes.ipynb) —
    compare multislice, full-grid, and ROI acceleration modes.
11. [`11_tilted_magnetic_layer_multislice.ipynb`](11_tilted_magnetic_layer_multislice.ipynb)
    — study tilted magnetic multilayers and fixed versus local wave-vector XMCD
    projection.
12. [`12_cobalt_l_edge_energy_sweep.ipynb`](12_cobalt_l_edge_energy_sweep.ipynb)
    — scan the Co L edges and compare energy-dependent XMCD exit waves,
    holograms, and reconstructions.
13. [`13_mumax_ovf_workflow.ipynb`](13_mumax_ovf_workflow.ipynb) — load a
    Mumax/OOMMF OVF magnetization stack and use it in a coherent-scattering
    simulation.
14. [`14_end_to_end_scattering_experiment.ipynb`](14_end_to_end_scattering_experiment.ipynb)
    — define a complete detector, sample, FTH mask, magnetic pattern, and CR/CL
    illumination; simulate exit waves, ideal and corrupted hologram sums and
    differences, and their FTH reconstructions.
15. [`15_camera_defects_and_cosmic_rays.ipynb`](15_camera_defects_and_cosmic_rays.ipynb)
    — model camera-persistent hot, cold, and flickering pixels alongside
    exposure-dependent cosmic-ray tracks with reproducible seeds.
16. [`16_compare_detector_propagation.ipynb`](16_compare_detector_propagation.ipynb)
    — propagate one shared FTH exit wave with Fraunhofer and Rayleigh–Sommerfeld;
    compare full holograms, half-image composites, signed differences, ratios,
    and radial averages versus scattering angle.

Every scattering notebook exposes `detector_propagation_method="fraunhofer"`
near the top. Choose `"rayleigh_sommerfeld"` for finite-distance propagation
after the sample; use small grids for this direct solver. Notebooks that stop
before detector propagation or only process supplied images document the
setting for extending their workflow. Detector and pipeline constructors pass
the selector explicitly, including the legacy examples. Tutorial 16 intentionally
runs both models to compare them. Ordinary pipeline runs keep RS disabled.

Notebooks 1–9 form the main introductory path. Notebooks 10–13 are advanced
or specialized and can be opened independently after the pipeline tutorial.
Tutorial 14 is the recommended end-to-end capstone and maintained replacement
for the old monolithic scattering notebooks. Tutorial 15 is a focused detector
artifact reference and can be run independently.

## Runnable scripts

- [`simulate_hologram_sweep.py`](simulate_hologram_sweep.py) creates balanced
  saturated/stripe-rich/bubble-rich hologram datasets and writes HDF5 output.
- [`train_ssl.py`](train_ssl.py) runs the configured self-supervised training
  workflows.
- [`evaluate_representations.py`](evaluate_representations.py) evaluates saved
  representations and produces metrics and plots.

## Legacy notebooks

Historical experiments that may still be useful for reference are in
[`legacy/`](legacy/README.md). They are not maintained as part of the numbered
tutorial path and may require adaptation before execution.

# FTH example API

Every maintained notebook uses `ExperimentConfig` and the same production
component classes. See the [standard setup guide](../../docs/experiment_setup.md)
for the complete editable template, units and lifecycle. Lengths are metres,
energies eV and angles radians; recipe thickness literals such as `Co(30)` are nm.
Arrays use `(y, x)`, and paper helicity pairs are ordered **CR, CL**.

```python
from scattering_calculator import simulation_pipelines as sim
from scattering_calculator.simulation_pipelines.examples import fth_experiment

experiment = fth_experiment()  # fills the standard sections with FTH defaults
experiment = experiment.with_changes(
    xray={"energy": 778.},
    sample={"max_slice_thickness": 5e-9},
    detector={"shape": (128, 128), "detector_center": (64, 64)},
)
run = sim.ScatteringExperiment(experiment).setup()
exit_wave = run.propagate().exit_wave
ideal_detector = run.detect()
measured_detector = run.detect(noise=True)
```

The ordered sections are `xray`, `simulation`, `sample`, `magnetic_pattern`,
`aperture`, `illumination`, `propagation`, `detector` and `beamstop`.
`energies_eV` specifies a scan; `analysis` holds postprocessing settings.
Use the same sections to build a different specimen, rather than defining a new
notebook-specific configuration class. `setup(magnetization=..., mask=...)`
accepts imported arrays on the configured grid.

| Change | Standard field |
| --- | --- |
| Photon energy / linear angle | `experiment.xray.energy` / `linear_polarization_angle` |
| Sample grid / pitch | `experiment.simulation.shape` / `real_space_pixel_size` |
| Layers / longitudinal discretization | `experiment.sample.recipe` / `max_slice_thickness` |
| Vector texture | `experiment.magnetic_pattern.pattern_type_method` / `pattern_config` |
| Object/reference geometry | `experiment.aperture.aperture_config` |
| Gaussian profile | `experiment.illumination.illumination_config` (`fwhm`, `center`, `distance`, `alpha_beam`) |
| Scalar / Jones / Stokes | `experiment.propagation.propagator_method` |
| Multislice / projection | `experiment.propagation.propagator_config["propagate"]` |
| FFT / finite-distance propagation | `experiment.detector.detector_propagation_method` |
| Pixel integration | `use_detector_pixel_footprint`, `detector_pixel_footprint_samples` in `detector` |
| Linear analyzer | `experiment.detector.analyzer_angle` (`None` removes it) |
| Sensor / acquisition | `detector_params`, `artifacts_config`, `measurement_config` in `detector` |
| Beamstop | `experiment.beamstop.bs_method` / `bs_config` |

`with_changes` returns an independent configuration. When changing energy,
construct a fresh run so optical constants, wavelength and camera q coordinates
all refresh. A spectral scan uses the same declaration at every energy:

```python
for energy in experiment.energies_eV:
    run = sim.ScatteringExperiment(experiment.with_changes(xray={"energy": energy}))
    image = run.run()
```

The bundled Co circular magnetic channel covers 770–805 eV and the bundled linear
channel is zero. Check the material database before selecting another edge.
Direct Rayleigh–Sommerfeld propagation scales with source pixels times detector
pixels; use a small grid for benchmarks. See the [propagation guide](../../docs/light_propagation_modes.md)
and [contrast guide](../../docs/optical_contrast_formalisms.md) for numerical options.

## Paper analysis helpers

The paper notebooks import `experiments` and `workflow` from this directory.
These functions consume the same configuration:

```python
case, ideal, measurement = wf.baseline(wf.OUT, experiment)
scan = experiment.with_changes(energies_eV=tuple(range(772, 801)))
wf.spectral(wf.OUT, scan)
metrics = wf.validate(wf.OUT, experiment)
```

`case['exits']` contains the CR/CL complex exit fields, `case['images']` their
sample-grid FFT intensities, and `case['reconstruction']` the complex inverse-FFT
correlation of their difference. The paper acquisition helper uses a common
synthetic display count budget for both helicities; it is not an absolute flux
calibration. Use `ScatteringExperiment.detect` for the production detector chain.

The scan saves common-q data and a separate fixed-camera series with per-energy
`qx`, `qy`. Regrid camera data before reconstructing on a common q grid.
The historical `sideband_rms` key stores root-sum-square sideband amplitude.
Validation reports representation agreement, slice refinement, projection
sensitivity and seeded acquisition repeatability. See [VALIDATION.md](VALIDATION.md).

Historical notebooks and documents remain in `paper/backups/` and `tutorials/legacy/`.

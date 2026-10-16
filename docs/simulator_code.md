# Reading the scattering simulator

Start with `src/scattering_calculator/simulation_pipelines/experiment.py`.
`ScatteringExperiment` connects the physical stages using ordinary method calls:

```python
from scattering_calculator.simulation_pipelines import ScatteringExperiment
from scattering_calculator.simulation_pipelines.examples import fth_experiment

experiment = fth_experiment()
run = ScatteringExperiment(experiment)
run.setup()                                    # sample and incident illumination
wavefront = run.propagate()                     # multislice → complex exit field
ideal = run.propagate_to_detector()             # free space → detector intensity
counts = run.acquire()                         # exposure and detector response
```

This example runs the source's single energy and polarization. For a scan and
HDF5 output use `simulate_experiment(experiment)` as described in the
[experiment setup guide](experiment_setup.md).

`setup()` lists the preparation steps explicitly: resolve the sample grid,
assign material occupancy and magnetization, build illumination, then connect the
propagator and detector. Supplied magnetization and occupancy arrays follow the
same path as generated inputs. Preparation runs on an independent copy of the
configuration.

| Physical operation | Implementation to read | Main choice |
| --- | --- | --- |
| Incident field | `experimental_conditions/light_beam.py`: `gauss_beam`, `illumination.plane_wave`; `IlluminationConfig.setup` scales total sampled power | Spatial envelope, phase, polarization and flux |
| Scalar material transmission | `beam_propagator/simple_propagation.py`: `scalar_wavefronts.apply_index_slice` | Effective complex refractive index |
| Jones material transmission | `beam_propagator/Jones_propagator.py`: `wavefronts.apply_eps_slice` | Projected dielectric tensor and polarization conversion |
| Mixed polarization | `beam_propagator/Stokes_propagator.py`: `stokes_wavefronts` | Coherent Jones carriers with incoherent intensity addition |
| Within-sample diffraction | `propagate_scalar_multislice`, `propagate_jones_multislice` in the corresponding propagator modules | Enabled/disabled diffraction; padding and region optimizations |
| Far-field detector propagation | `DetectorConfig.detect_hologram`, then `experimental_conditions/detector.py`: `detector_hologram.gnomonic_projection` | Fraunhofer FFT, flat-detector mapping and pixel footprints |
| Finite-distance detector propagation | `DetectorConfig._detect_rayleigh_sommerfeld`, then `beam_propagator/detector_propagation.py`: `RayleighSommerfeldPropagator.forward` | Direct scalar integration at detector coordinates |
| Detector signal | `experimental_conditions/detector.py`: `detector_hologram.add_noise` | Exposure, efficiency, photon statistics and sensor artifacts |

`DetectorConfig`, `IlluminationConfig` and `SamplePropagatorConfig` live in
`simulation_pipelines/simulation_configuration.py`. They connect configuration
values to the numerical functions; the propagation equations live in the
propagator modules. The [contrast guide](optical_contrast_formalisms.md) and
[propagation guide](light_propagation_modes.md) explain their equations and
approximations.

In each multislice loop, material transmission comes first. The callback, when
supplied, observes that post-material field. Diffraction then advances it toward
the next slice. The loop ends immediately after the final material interaction;
propagation from that exit plane to the detector is a separate stage.

`propagate_to_detector()` returns ideal intensity. `acquire()` reuses it and
applies detector effects without rerunning sample or free-space propagation.
`detect(noise=False)` and `detect(noise=True)` remain the combined convenience
calls. An analyzer acts on complex amplitudes before intensities are formed;
incoherent polarization or spectral components are summed as intensities before
acquisition.

# `fo-mo-cid`

All maintained scattering examples share the [standard experiment setup](experiment_setup.md). Use its component sections, units and execution stages for new notebooks.

`fo-mo-cid` documents and ships tutorial-first workflows for self-supervised
learning and coherent-scattering simulation.

- `CIFAR-10` with `MAE`
- `STL-10` with `DINO`
- Fourier-transform holography (FTH) coherent-scattering simulations

The repository is organized so each surface has a clear role:

- `README.md` is the onboarding entrypoint
- `docs/` is the canonical prose and API reference
- `tutorials/` contains runnable CLIs and focused notebooks
- `configs/` contains shipped example and smoke-test YAML files

Start here:

- Read the [tutorial guide](tutorials.md)
- Use the [coherent-scattering tutorial map](coherent_scattering_tutorials.md)
  for FTH simulation notebooks and pipeline usage
- Follow the [simulator source walkthrough](simulator_code.md) to trace the four
  physical stages in the code
- See [illumination and coherence](illumination.md) for beam profiles, polarization, simultaneous colors and separate scans
- Use the [material dataset guide](material_datasets.md) for energy-dependent optical channels
- Read the [detector response guide](detector_response.md) for energy-dependent ADU gain and nonlinear calibration
- Read the [optical contrast formalisms](optical_contrast_formalisms.md) for
  the scalar refractive-index model, Jones dielectric tensors, XMCD/XMLD
  equations, and vector contrast with local light momentum
- Read the [light propagation modes](light_propagation_modes.md) for the
  no-propagation approximation, multislice FFT propagation, final Fraunhofer
  propagation, and detector q-space projection
- Read the [hologram sweep reference](simulate_hologram_sweep.md) for scripted
  HDF5 dataset generation
- Share the [magnetic-pattern analysis guide](magnetic_pattern_analysis.md) for
  HDF5 loading, units, measured states and analysis examples
- Use the [configuration reference](configuration.md) when editing YAML configs
- Browse the [API reference](api.rst) for the reusable package surface
- See the [roadmap](roadmap.md) for planned extensions

```{toctree}
:maxdepth: 2
:caption: Contents

configuration
tutorials
coherent_scattering_tutorials
simulator_code
illumination
material_datasets
detector_response
optical_contrast_formalisms
light_propagation_modes
simulate_hologram_sweep
magnetic_pattern_analysis
roadmap
contributing
api
```

- [General magnetic domains and measured parameter atlas](magnetic_domain_catalogue.md)

- [Experimental constrained dipolar domains](constrained_magnetic_domains.md)

- [Hierarchical hologram sweeps with stage reuse](hierarchical_hologram_sweep.md)

- [Single-experiment Python desktop GUI](single_experiment_gui.md)

- [HDF5 save modes: LZF, gzip, or uncompressed](hdf5_save_modes.md)

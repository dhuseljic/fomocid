# `fo-mo-cid`

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
- Read the [hologram sweep reference](simulate_hologram_sweep.md) for scripted
  HDF5 dataset generation
- Use the [configuration reference](configuration.md) when editing YAML configs
- Browse the [API reference](api.rst) for the reusable package surface
- See the [roadmap](roadmap.md) for planned extensions

```{toctree}
:maxdepth: 2
:caption: Contents

configuration
tutorials
coherent_scattering_tutorials
simulate_hologram_sweep
roadmap
contributing
api
```

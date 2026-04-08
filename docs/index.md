# `fo-mo-cid`

`fo-mo-cid` documents and ships two tutorial-first self-supervised learning workflows on standard image benchmarks:

- `CIFAR-10` with `MAE`
- `STL-10` with `DINO`

The repository is organized so each surface has a clear role:

- `README.md` is the onboarding entrypoint
- `docs/` is the canonical prose and API reference
- `tutorials/` contains the runnable CLIs
- `configs/` contains shipped example and smoke-test YAML files

Start here:

- Read the [tutorial guide](tutorials.md)
- Use the [configuration reference](configuration.md) when editing YAML configs
- Browse the [API reference](api.rst) for the reusable package surface
- See the [roadmap](roadmap.md) for coherent-imaging work that is intentionally deferred

```{toctree}
:maxdepth: 2
:caption: Contents

configuration
tutorials
roadmap
contributing
api
```

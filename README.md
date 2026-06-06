# fo-mo-cid

`fo-mo-cid` is a tutorial-first repository for self-supervised learning and coherent-scattering simulation workflows. The current shipped scope includes:

- `CIFAR-10` with `MAE`
- `STL-10` with `DINO`
- coherent FTH hologram simulation notebooks and sweep-generation utilities

The repository keeps notebooks, scripts, configs, and the shared `src/fomocid` package aligned so the examples are both runnable and documented.

## Install

```bash
conda activate fo-mo-cid
pip install -e .
```

To build the docs locally too:

```bash
pip install -e ".[docs]"
```

## Quickstart

Open the canonical tutorial notebooks:

- [MAE pretraining notebook](docs/notebook_tutorials/01_mae_pretraining.ipynb)
- [MAE analysis notebook](docs/notebook_tutorials/02_mae_analysis.ipynb)
- [Coherent-scattering tutorial map](docs/coherent_scattering_tutorials.md)

Inspect the config contract from the CLI:

```bash
python tutorials/train_ssl.py --help
python tutorials/train_ssl.py --config configs/cifar10_mae.yaml --output-dir outputs --explain-config
```

Run CIFAR-10 MAE pretraining:

```bash
python tutorials/train_ssl.py --config configs/cifar10_mae.yaml --output-dir outputs
```

Run the fastest real-dataset MAE example:

```bash
python tutorials/train_ssl.py --config configs/mnist_mae.yaml --output-dir outputs
```

Run CIFAR-10 evaluation:

```bash
python tutorials/evaluate_representations.py \
  --config configs/cifar10_mae.yaml \
  --checkpoint-path outputs/<run>/checkpoints/last.ckpt \
  --output-dir outputs
```

Run STL-10 DINO pretraining:

```bash
python tutorials/train_ssl.py --config configs/stl10_dino.yaml --output-dir outputs
```

Run STL-10 evaluation:

```bash
python tutorials/evaluate_representations.py \
  --config configs/stl10_dino.yaml \
  --checkpoint-path outputs/<run>/checkpoints/last.ckpt \
  --output-dir outputs
```

Run a coherent FTH hologram sweep:

```bash
python tutorials/simulate_hologram_sweep.py
```

For the notebook-first path, start with [the coherent-scattering tutorial map](docs/coherent_scattering_tutorials.md), then open the focused notebooks under `tutorials/`.
For micromagnetic inputs, use `tutorials/Scattering_simulator_CK_mumax.ipynb`;
it reads Mumax/OOMMF OVF files from `DATA_ROOT/Data/mumax_files/`, checks the
recipe against the OVF `z` stack, and shows exit-wave and FTH reconstruction
diagnostics.

## Repository Layout

```text
.
├── configs/                # Shipped tutorial and smoke-test YAML configs
├── docs/                   # Canonical docs, config reference, and notebooks
├── src/fomocid/            # Shared package for data, SSL, eval, analysis, utils
├── tests/                  # Smoke tests and repository contract checks
└── tutorials/              # Runnable CLIs and coherent-scattering notebooks
```

This is a source-only repository. Built docs, checkpoints, cached artifacts, `egg-info`, and other generated files are not part of the committed project state.

## Documentation Map

- [Docs landing page](docs/index.md)
- [Tutorial guide](docs/tutorials.md)
- [Coherent-scattering tutorials](docs/coherent_scattering_tutorials.md)
- [Hologram sweep documentation](docs/simulate_hologram_sweep.md)
- [Configuration reference](docs/configuration.md)
- [Contribution notes](docs/contributing.md)
- [Roadmap](docs/roadmap.md)

## Notes

- The notebooks are the main teaching surface, and the scripts in `tutorials/` are the executable source of truth underneath them.
- Build the docs site with `python -m sphinx -W --keep-going -b html docs docs/_build/html`.
- Smoke configs use fake data so CI does not depend on benchmark downloads.
- `configs/mnist_mae.yaml` is the lightest real-dataset MAE example in the repo.
- Coherent-scattering notebooks are intentionally focused by topic. Use the map in [docs/coherent_scattering_tutorials.md](docs/coherent_scattering_tutorials.md) to pick the smallest notebook for the concept you want to learn.

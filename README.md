# fo-mo-cid

`fo-mo-cid` is a tutorial-first repository for self-supervised learning on standard image benchmarks. The current shipped scope is two reproducible phase-1 workflows:

- `CIFAR-10` with `MAE`
- `STL-10` with `DINO`

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

## Repository Layout

```text
.
├── configs/                # Shipped tutorial and smoke-test YAML configs
├── docs/                   # Canonical docs, config reference, and notebooks
├── src/fomocid/            # Shared package for data, SSL, eval, analysis, utils
├── tests/                  # Smoke tests and repository contract checks
└── tutorials/              # Runnable training and evaluation CLIs
```

This is a source-only repository. Built docs, checkpoints, cached artifacts, `egg-info`, and other generated files are not part of the committed project state.

## Documentation Map

- [Docs landing page](docs/index.md)
- [Tutorial guide](docs/tutorials.md)
- [Configuration reference](docs/configuration.md)
- [Contribution notes](docs/contributing.md)
- [Roadmap](docs/roadmap.md)

## Notes

- The notebooks are the main teaching surface, and the scripts in `tutorials/` are the executable source of truth underneath them.
- Build the docs site with `python -m sphinx -W --keep-going -b html docs docs/_build/html`.
- Smoke configs use fake data so CI does not depend on benchmark downloads.
- `configs/mnist_mae.yaml` is the lightest real-dataset MAE example in the repo.
- Future coherent-imaging work is documented in [docs/roadmap.md](docs/roadmap.md), not shipped as active package functionality yet.

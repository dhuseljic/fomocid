# Tutorials

The tutorial notebooks are the main teaching surface, and the scripts under `tutorials/` are the executable source of truth.

## Supported workflows

- `configs/cifar10_mae.yaml` trains and evaluates a CIFAR-10 MAE tutorial run.
- `configs/mnist_mae.yaml` trains a smaller, faster MAE example on MNIST.
- `configs/stl10_dino.yaml` trains and evaluates an STL-10 DINO tutorial run.

## Useful CLI entrypoints

```bash
python tutorials/train_ssl.py --help
python tutorials/train_ssl.py --config configs/cifar10_mae.yaml --output-dir outputs --explain-config
python tutorials/evaluate_representations.py --help
```

## Notebook tutorials

```{toctree}
:maxdepth: 1
:caption: Notebook Tutorials

notebook_tutorials/01_mae_pretraining
notebook_tutorials/02_mae_analysis
```

## Script entrypoints

- `tutorials/train_ssl.py` runs pretraining and writes `resolved_config.yaml`, `checkpoints/`, optional `logs/`, and `train_summary.json`.
- `tutorials/evaluate_representations.py` writes `embeddings.pt`, `metrics.json`, `projection.png`, `nearest_neighbors.png`, and `evaluation_summary.json`.

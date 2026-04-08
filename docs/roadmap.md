# Roadmap

## Phase 1: Learn the workflow on standard image benchmarks

- Build a stable MAE tutorial on CIFAR-10 and a stable DINO tutorial on STL-10.
- Keep training, evaluation, and analysis script-backed and reproducible.
- Use a small shared package only where it removes duplication between tutorials.

## Phase 2: Introduce coherent-imaging-specific data handling

- Add raw coherent imaging datasets and preprocessing abstractions.
- Revisit augmentations for complex-valued or amplitude/phase data.
- Extend evaluation protocols beyond classification-centric probes.

## Phase 3: Foundation-model training workflows

- Scale from tutorial models toward larger backbones and larger datasets.
- Add checkpoint management, experiment tracking, and richer distributed training support.
- Define domain-specific downstream tasks for reconstruction, retrieval, and scientific discovery.

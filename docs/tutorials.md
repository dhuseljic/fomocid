# Tutorials

The tutorial notebooks are the main teaching surface, and the scripts under
`tutorials/` are the executable source of truth.

## Supported workflows

- `configs/cifar10_mae.yaml` trains and evaluates a CIFAR-10 MAE tutorial run.
- `configs/mnist_mae.yaml` trains a smaller, faster MAE example on MNIST.
- `configs/stl10_dino.yaml` trains and evaluates an STL-10 DINO tutorial run.
- `tutorials/simulate_hologram_sweep.py` generates coherent FTH hologram
  sweeps and writes HDF5 datasets.

## Useful CLI entrypoints

```bash
python tutorials/train_ssl.py --help
python tutorials/train_ssl.py --config configs/cifar10_mae.yaml --output-dir outputs --explain-config
python tutorials/evaluate_representations.py --help
python tutorials/simulate_hologram_sweep.py
```

## Notebook tutorials

```{toctree}
:maxdepth: 1
:caption: Notebook Tutorials

notebook_tutorials/01_mae_pretraining
notebook_tutorials/02_mae_analysis
```

## Coherent-scattering notebooks

The coherent-scattering notebooks are organized by topic so users can learn one
part of the simulation at a time:

- [`tutorial_beamstop_definition.ipynb`](../tutorials/tutorial_beamstop_definition.ipynb)
- [`tutorial_material_holography_mask_definition.ipynb`](../tutorials/tutorial_material_holography_mask_definition.ipynb)
- [`tutorial_illumination_function_definition.ipynb`](../tutorials/tutorial_illumination_function_definition.ipynb)
- [`tutorial_magnetic_pattern_definition.ipynb`](../tutorials/tutorial_magnetic_pattern_definition.ipynb)
- [`tutorial_material_parameters_definition.ipynb`](../tutorials/tutorial_material_parameters_definition.ipynb)
- [`tutorial_hologram_generation_and_artifacts.ipynb`](../tutorials/tutorial_hologram_generation_and_artifacts.ipynb)
- [`tutorial_hologram_pipeline_usage.ipynb`](../tutorials/tutorial_hologram_pipeline_usage.ipynb)
- [`tutorial_multislice_and_roi_modes.ipynb`](../tutorials/tutorial_multislice_and_roi_modes.ipynb)
- [`tutorial_tilted_magnetic_layer_multislice.ipynb`](../tutorials/tutorial_tilted_magnetic_layer_multislice.ipynb)
- [`tutorial_cobalt_l_edge_energy_sweep.ipynb`](../tutorials/tutorial_cobalt_l_edge_energy_sweep.ipynb)
- [`Scattering_simulator_CK_mumax.ipynb`](../tutorials/Scattering_simulator_CK_mumax.ipynb)

Use the [coherent-scattering tutorial map](coherent_scattering_tutorials.md)
to choose the right notebook, and the
[hologram sweep reference](simulate_hologram_sweep.md) for the scripted
pipeline and HDF5 output layout.
Use the [optical contrast formalisms](optical_contrast_formalisms.md) chapter
for the scalar refractive-index equations, Jones dielectric tensors, charge,
XMCD, XMLD, and vector/local-momentum contrast.
Use the [light propagation modes](light_propagation_modes.md) chapter for
no-propagation, multislice propagation, final far-field FFTs, and detector
q-space projection.

The tilted magnetic layer notebook is the focused reference for vector XMCD
contrast: it compares a fixed beam-direction `m . k` projection with the
local-k mode that estimates the light momentum from Jones phase gradients
during multislice propagation.

The Mumax notebook is the path for CK-style single simulations driven by a
Mumax/OOMMF OVF magnetization file. It memory-maps the OVF data, checks that
the material recipe has one magnetic propagated layer per OVF `z` cell, and
visualizes CL-CR exit waves plus ideal/detected FTH difference reconstructions.

## Script entrypoints

- `tutorials/train_ssl.py` runs pretraining and writes `resolved_config.yaml`, `checkpoints/`, optional `logs/`, and `train_summary.json`.
- `tutorials/evaluate_representations.py` writes `embeddings.pt`, `metrics.json`, `projection.png`, `nearest_neighbors.png`, and `evaluation_summary.json`.
- `tutorials/simulate_hologram_sweep.py` writes a single HDF5 file with one
  group per synthetic coherent-scattering experiment, including CR/CL
  holograms, exit waves, masks, magnetic pattern crops, and metadata.

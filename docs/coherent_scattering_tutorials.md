# Coherent-Scattering Tutorials

The coherent-scattering notebooks in `tutorials/` are split by topic so users
do not need to start from the full scattering notebook when they only want to
understand one part of the simulation.

## Where To Start

For a first pass:

1. Open the focused notebook for the concept you want to learn.
2. Run only the parameter and visualization cells first.
3. Move to the pipeline notebooks when you want scripted sweeps and HDF5 output.

The full CK notebook remains useful as an end-to-end interactive reference, but
the notebooks below are the recommended teaching path.

Its final export section uses `HologramPipeline.build_precomputed_metadata()`
and `HologramPipeline.write_precomputed_result()` so an interactively simulated
result is saved with the same canonical HDF5 structure as a scripted sweep.

## Focused Notebooks

| Topic | Notebook | What it teaches |
|---|---|---|
| Beamstop definition | [`tutorial_beamstop_definition.ipynb`](../tutorials/tutorial_beamstop_definition.ipynb) | Detector geometry, physical beamstop parameters, projected mask, wires, roughness, anti-aliasing, and line-profile checks. |
| Holography mask definition | [`tutorial_material_holography_mask_definition.ipynb`](../tutorials/tutorial_material_holography_mask_definition.ipynb) | FTH object/reference holes, 3-D aperture masks, layer-by-layer views, vertical cuts, and support masks. |
| Illumination function | [`tutorial_illumination_function_definition.ipynb`](../tutorials/tutorial_illumination_function_definition.ipynb) | Gaussian and plane-wave illumination, beam centre/FWHM/focus distance, Jones polarization, and RGB complex-field visualization. |
| Magnetic pattern | [`tutorial_magnetic_pattern_definition.ipynb`](../tutorials/tutorial_magnetic_pattern_definition.ipynb) | Labyrinth, stripe, skyrmion, and saturated patterns; physical `sigma` smoothing; statistics; and vector magnetization mapping. |
| Material parameters | [`tutorial_material_parameters_definition.ipynb`](../tutorials/tutorial_material_parameters_definition.ipynb) | Recipe parsing, effective layers, refractive-index channels, material stack visualization, and compact dielectric tensors. |
| Hologram generation and artifacts | [`tutorial_hologram_generation_and_artifacts.ipynb`](../tutorials/tutorial_hologram_generation_and_artifacts.ipynb) | Ideal/detected holograms, detector artifacts, beamstop/noise effects, CR-CL/CR+CL channels, and FTH reconstruction. |
| Pipeline usage | [`tutorial_hologram_pipeline_usage.ipynb`](../tutorials/tutorial_hologram_pipeline_usage.ipynb) | `HologramPipelineConfig`, `HologramPipelineRanges`, running sweeps, HDF5 layout, reading outputs, masks, magnetic patterns, and CR/CL exit waves. |
| Multislice and ROI modes | [`tutorial_multislice_and_roi_modes.ipynb`](../tutorials/tutorial_multislice_and_roi_modes.ipynb) | `propagate=True/False`, aperture ROI modes, Jones/tensor ROI, multislice ROI, padding/absorbers, and practical mode choices. |

## Scripted Sweep

Use [`tutorials/simulate_hologram_sweep.py`](../tutorials/simulate_hologram_sweep.py)
when you want to generate a dataset rather than interactively explore one
simulation. The script uses:

- `HologramPipelineConfig` for fixed/default parameters;
- `HologramPipelineRanges` for sampled parameters;
- `HologramPipeline.run()` to write one HDF5 file with one group per simulated
  sample.

The prose reference for the sweep script is
[`docs/simulate_hologram_sweep.md`](simulate_hologram_sweep.md).

## Choosing ROI And Multislice Modes

The practical mode choices are:

- **Jones-only with aperture ROIs**: `propagate=False`, `use_roi=True`,
  `dielectric_tensor_use_roi=True`, `dielectric_tensor_compact=True`.
  This is the fastest default for thin stacks.
- **Full-field multislice**: `propagate=True`,
  `multislice_propagation_roi=False`. This keeps free-space propagation
  physically conservative because the FFT is applied to the full field.
- **Multislice with Jones/tensor ROI only**: `propagate=True`, `use_roi=True`,
  `dielectric_tensor_use_roi=True`, `multislice_propagation_roi=False`.
  This reduces local tensor/Jones work while keeping full-field free-space FFTs.
- **ROI multislice, default merged crops**: `propagate=True`, `use_roi=True`,
  `dielectric_tensor_use_roi=True`, `multislice_propagation_roi=True`,
  `multislice_propagation_roi_merge_overlaps=True`. This is fastest for large
  sweeps when the aperture regions are small, and overlapping padded crops are
  treated as one local propagation problem.
- **ROI multislice, separate-crop opt-out**:
  `multislice_propagation_roi_merge_overlaps=False`. This can be faster when
  padded aperture boxes are well separated, but validate it against the merged
  default or a full-field reference if close OH/RH layouts are involved.
- **Vertical x-z aperture diagnostics**: the multislice/ROI tutorial includes a
  didactic x-z wavefront view through each aperture for every propagation mode:
  Jones-only, full-field multislice, Jones/tensor ROI with full multislice, ROI
  multislice with separate crops, and ROI multislice with merged crops. It
  rebuilds the tutorial aperture stack, overlays aperture walls plus SiN and
  magnetic-layer borders, marks the 1-D ROI propagation intervals for ROI modes,
  and shows amplitude/phase through the stack. This is a visualization aid, not
  an extra HDF5 output generated by the pipeline.
- **Full-field reference/debug mode**: `use_roi=False`. Use this for small
  arrays or reference comparisons.

These ROI modes are computational approximations, not hard masks. Outside an
ROI, the solver still carries a field forward:

- `magnetic_pattern_use_roi=True` initializes the full magnetic-pattern plane to
  the background value `+1`, generates supported expensive textures only in a
  padded object-hole box, and pastes that texture into the full plane. Outside
  the box, the magnetic texture remains background.
- `dielectric_tensor_use_roi=True` computes dense magnetic/vacuum dielectric
  corrections only in aperture support boxes. Outside those boxes, each layer is
  treated as the spatially uniform diagonal background response, so Jones
  propagation still applies the layer transmission there.
- `multislice_propagation_roi=True` runs the inter-slice FFT propagator only in
  padded aperture boxes. The solver starts from the zero-spatial-frequency
  plane-wave phase everywhere, then adds each ROI crop's local diffraction
  correction relative to that baseline. By default,
  `multislice_propagation_roi_merge_overlaps=True`, so overlapping padded boxes
  are merged before propagation and nearby apertures are treated as one local
  diffraction problem instead of competing in shared pixels. Set it to `False`
  for the faster separate-crop approximation. Pixels outside all boxes keep only
  the plane-wave phase, not a full diffraction calculation.

Use full-field multislice, `propagate=True` with
`multislice_propagation_roi=False`, when diffraction between ROI and non-ROI
pixels is important.

For a deeper explanation and runnable comparisons, use
[`tutorial_multislice_and_roi_modes.ipynb`](../tutorials/tutorial_multislice_and_roi_modes.ipynb).

## Reading HDF5 Outputs

Pipeline outputs contain a `_pipeline_config/` group plus one numbered group
per simulated sample:

```text
simulation_sweep.h5
├── _pipeline_config/
├── 00000/
│   ├── CR/
│   ├── CL/
│   ├── beamstop_mask
│   ├── supportmask
│   ├── magnetic_pattern_oh
│   └── metadata/
└── ...
```

Detector acquisition metadata is separated into sibling
`measurement_config/`, `detector_params/`, and `artifacts_config/` groups.
`detector_params/counts_per_photon` is the single canonical counts-to-photon
conversion setting; it is not duplicated under artifact settings or
`_pipeline_config/`. Sweepable configuration dictionaries are saved once per
numbered sample so the stored values are always the values actually used.

Sample-owned settings are grouped under `metadata/sample/`, including the
master `use_roi` switch, aperture geometry, magnetic-pattern settings, and
dielectric-tensor settings.
All propagation metadata is grouped under `metadata/propagator_config/`, with
`propagator_method` first and no separate propagation group.

The complete annotated HDF5 tree and canonical path table are in
[`simulate_hologram_sweep.md`](simulate_hologram_sweep.md#hdf5-layout).

The pipeline usage notebook shows how to load:

- `CR/ideal`, `CL/ideal`;
- `CR/detected`, `CL/detected`;
- optional `CR/detected_no_beamstop`, `CL/detected_no_beamstop`;
- complex `CR/exit_wave`, `CL/exit_wave`;
- `beamstop_mask`, `supportmask`, and `magnetic_pattern_oh`;
- scalar and array metadata.

See [`tutorial_hologram_pipeline_usage.ipynb`](../tutorials/tutorial_hologram_pipeline_usage.ipynb)
for plotting CR/CL, CR-CL, masks, magnetic patterns, and exit waves.

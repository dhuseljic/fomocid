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
| Illumination function | [`tutorial_illumination_function_definition.ipynb`](../tutorials/tutorial_illumination_function_definition.ipynb) | Gaussian and plane-wave illumination, beam centre/FWHM/focus distance, optional beam tilt, Jones polarization, and RGB complex-field visualization. |
| Magnetic pattern | [`tutorial_magnetic_pattern_definition.ipynb`](../tutorials/tutorial_magnetic_pattern_definition.ipynb) | Labyrinth, stripe, skyrmion, and saturated patterns; physical `sigma` smoothing; statistics; and vector magnetization mapping. |
| Material parameters | [`tutorial_material_parameters_definition.ipynb`](../tutorials/tutorial_material_parameters_definition.ipynb) | Recipe parsing, effective layers, refractive-index channels, material stack visualization, and compact dielectric tensors. |
| Hologram generation and artifacts | [`tutorial_hologram_generation_and_artifacts.ipynb`](../tutorials/tutorial_hologram_generation_and_artifacts.ipynb) | Ideal/detected holograms, detector artifacts, beamstop/noise effects, CR-CL/CR+CL channels, and FTH reconstruction. |
| Pipeline usage | [`tutorial_hologram_pipeline_usage.ipynb`](../tutorials/tutorial_hologram_pipeline_usage.ipynb) | `HologramPipelineConfig`, `HologramPipelineRanges`, running sweeps, HDF5 layout, reading outputs, masks, magnetic patterns, and CR/CL exit waves. |
| Multislice and ROI modes | [`tutorial_multislice_and_roi_modes.ipynb`](../tutorials/tutorial_multislice_and_roi_modes.ipynb) | `propagate=True/False`, aperture ROI modes, Jones/tensor ROI, multislice ROI, padding/absorbers, and practical mode choices. |
| CK workflow with Mumax OVF input | [`Scattering_simulator_CK_mumax.ipynb`](../tutorials/Scattering_simulator_CK_mumax.ipynb) | CK-style single-simulation workflow where the magnetic layer count and 3-D magnetization stack come from a memory-mapped Mumax/OOMMF `.ovf` file in `DATA_ROOT/Data/mumax_files/`. Includes recipe compatibility checks, Mumax-to-sample interpolation, CL-CR exit-wave visualization, and ideal/detected FTH difference reconstructions. |

## Mumax OVF Workflow

[`Scattering_simulator_CK_mumax.ipynb`](../tutorials/Scattering_simulator_CK_mumax.ipynb)
is the CK-style notebook for micromagnetic input. It uses
`scattering_calculator.utils.mumax.read_mumax_ovf(..., mmap=True)` so the
header and binary vector field are exposed quickly as a lazy array shaped
`(znodes, ynodes, xnodes, 3)`.

The notebook expects the user to set `MUMAX_FILE_NAME` in the OVF reader cell.
The file is read from `DATA_ROOT / "Data" / "mumax_files"`. Folder listing is
optional (`LIST_AVAILABLE_OVF=True`) because scanning large external data
folders can be slower than opening the OVF header itself.

The sample recipe must contain exactly one propagated magnetic layer per Mumax
`z` cell. For example, if the OVF header reports `znodes=43`, the recipe must
produce 43 magneto-optic layers that can receive those slices. The notebook
prints the OVF-to-sample layer mapping and raises a clear error when the recipe
and OVF stack are incompatible.

The lateral Mumax field is treated as periodic. During interpolation the OVF
unit cell is wrapped in `x` and `y`, so a Mumax simulation smaller than the
object hole is tiled across the full scattering grid instead of being padded
with zero magnetization.

The output inspection section uses the current `HologramConfig` stores:
`exit_waves`, `ideal_holograms`, `detected_holograms`, and
`reconstructions`. It plots CR/CL exit-wave amplitude and phase, the direct
`CL - CR` complex exit-wave difference, ideal/detected CR-CL and CR+CL
holograms, and FTH reconstructions of the ideal and detected CR-CL difference
channels.

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
  This is the fastest default for thin stacks. With the default
  `jones_apply_zero_order_phase=True`, it still keeps the rank-zero
  inter-layer free-space phase `exp(-1j * k0 * dz)` while skipping transverse
  FFT diffraction.
- **Scalar eigenmode interaction**: set `propagator_method="Scalar"` to replace
  the two-component Jones interaction with a scalar refractive-index
  interaction. It skips dielectric-tensor construction, uses the database
  channels `[n_total, n_circ, n_lin]` directly with the aperture mask and
  magnetization to build `final_scalar_refractive_index`, and propagates one
  scalar 2-D field for the selected polarization. It is exact relative to Jones
  when that polarization is a local eigenvector of the interaction matrix. Keep
  `scalar_refractive_index_lazy=True` to compute scalar ROI patches layer by
  layer during propagation instead of precomputing the whole compact scalar
  stack. Scalar and Jones use the same `exp(-i kz dz)` free-space propagation
  convention in both full-field and ROI multislice modes. Keep
  `propagator_method="Jones"` when polarization mixing/rotation is physically
  important.
- **Full-field multislice**: `propagate=True`,
  `multislice_propagation_roi=False`. This keeps free-space propagation
  physically conservative because the FFT is applied to the full field.
- **Multislice with Jones/tensor ROI only**: `propagate=True`, `use_roi=True`,
  `dielectric_tensor_use_roi=True`, `multislice_propagation_roi=False`.
  This reduces local tensor/Jones work while keeping full-field free-space FFTs.
- **ROI multislice, default common crop**: `propagate=True`, `use_roi=True`,
  `dielectric_tensor_use_roi=True`, `multislice_propagation_roi=True`,
  `multislice_propagation_roi_merge_overlaps=True`. This uses one common crop
  enclosing all padded aperture boxes, so apertures can diffract into each
  other inside the local ROI without paying for a full-field FFT.
- **ROI multislice, separate-crop opt-out**:
  `multislice_propagation_roi_merge_overlaps=False`. This can be faster when
  padded aperture boxes are well separated. Intersecting aperture funnels are
  forced into one physical ROI before padding, and overlapping padded crops are
  also auto-merged to avoid double-adding local diffraction corrections.
  Validate this mode against the merged default or a full-field reference if
  close OH/RH layouts are involved.
- **Vertical x-z aperture diagnostics**: the multislice/ROI tutorial includes a
  didactic x-z wavefront view through each aperture for every propagation mode:
  Jones-only, full-field multislice, Jones/tensor ROI with full multislice, ROI
  multislice with separate crops, and ROI multislice with a common crop. It
  rebuilds the tutorial aperture stack, uses tight crops around each top
  aperture opening, marks the aperture wall, and shows amplitude/phase through
  the stack. This is a visualization aid, not an extra HDF5 output generated by
  the pipeline.
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
  correction relative to that baseline. Jones and Scalar use the same
  angular-spectrum sign convention: local crops use `exp(-i kz dz)`, and the
  outside baseline is the `kz = k0` limit of that operator. With
  `multislice_propagation_roi_merge_overlaps=True`, all padded aperture boxes
  are enclosed in one common crop. With the flag set to `False`, disjoint
  padded boxes may remain separate for speed. Aperture funnels that overlap in
  the actual material mask are always merged into one ROI, regardless of the
  flag, because they are one physical hole system. Padded boxes that overlap are
  also merged before propagation, even when the flag is `False`, because
  otherwise their local corrections would be added more than once in shared
  pixels. Each local correction is smoothly tapered to zero across the ROI
  padding before it is added back to the full field, which suppresses
  rectangular crop-edge bands. Set the flag to `False` only for the faster
  separate-crop approximation between disjoint padded boxes. Pixels outside all
  boxes keep only the plane-wave phase, not a full diffraction calculation.

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

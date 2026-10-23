# Reusing simulation stages

Run the new example from the repository root with the project environment active:

```bash
PYTHONPATH=src python tutorials/simulate_hologram_tree.py --patterns 5
```

By default results are saved to `DATA_ROOT / "Data" / "hologram_tree" /
"simulation_tree.h5"`, following the sweep data-folder convention. Override
this with `--output /path/to/file.h5` if needed.

The example creates **30 CR/CL acquisition pairs**: five magnetic patterns × two
exit-wave variants × one fixed detector geometry × three acquisitions. Edit the
`exits`, `geometries` and `acquisitions` lists in that script to change branching.
`--patterns` counts independent magnetic parents, not final output groups.
The example keeps detector distance fixed at 40 mm for every branch. At the
parent energy of 778 eV and 20 um detector pitch, the real-space field is about
3.19 um wide, containing the 600 nm OH and reference holes at 1 um offsets.
The earlier 5 mm setting gave only 0.40 um and clipped the mask; increasing
oversampling alone cannot enlarge that field. Distance variation remains an
optional API capability, but the example does not use it.

The original `simulate_hologram_sweep.py` remains available for independent runs.
The output path is overwritten when the pipeline runs, like the original sweep.

## What is reused

| Stage | Recomputed for | Reused by |
| --- | --- | --- |
| Binary magnetic texture and physical domain-wall smoothing | Each independent pattern parent | All its exit-wave variants |
| Sample optical response, aperture mask and exit waves, including their far-field representation | Each exit variant, separately for CR and CL | All camera geometries and acquisitions below it |
| Ideal detector projection | Each geometry and polarization | All acquisitions below it |
| Photon statistics, charge spreading, response, readout, defects and ADC | Each acquisition and polarization | No stochastic exposure is reused |

Sample construction and magnetic state measurements currently run once per exit
variant. The magnetic texture itself is generated once per parent. Memory caches
hold only the current parent's texture, current exit-wave pair and current ideal
pair; they are cleared as their parent changes. Output uses the existing flat
HDF5 layout and writes shared arrays into each leaf for compatibility, so disk
space and compression time still grow with the number of acquisitions.

## Parameter dependencies

`HierarchicalHologramPipeline` accepts the usual `config`, `ranges`,
`output_path`, `n_samples` and `verbose`, plus three nonempty lists of explicit
variant dictionaries. Ranges are sampled once per magnetic parent. Variant
values override those sampled values; omitted values keep the parent settings.
Variants are independent overrides, not cumulative changes to the previous one.

* `exit_variants` uses `HologramPipelineConfig` names: `xray_energy` and
  `aperture_radii`, `aperture_centers`, `aperture_sigmas`, `aperture_lengths`,
  `aperture_angles`, `aperture_ellipticities`, `aperture_top_radius_factors`.
  Aperture lists retain their size and OH entries; change RH entries to explore
  reference-mask designs. Changing OH geometry requires a separate parent/run.
  Energy changes recompute material response and sample propagation.
* `geometry_variants` uses `DetectorConfig` names:
  `sample_to_detector_distance`, `pixel_size`, `detector_center`,
  `use_detector_pixel_footprint`, `detector_pixel_footprint_samples`,
  `ignore_flat_detector_curvature`, `detector_propagation_method`.
  These recalculate detector coordinates, beamstop and support mask. Finite-distance
  propagation, when selected, is recalculated from the cached field.
* `acquisition_variants` contains `measurement_config`, `artifacts_config`, and
  `detector_params` dictionaries. Each merges with its parent dictionary.
  Examples include exposure seconds, `number_frames`, `acquisition_mode`,
  quantum efficiency, readout noise, charge spreading, defects, cosmic rays,
  gain and optional nonlinear response. Linear response and 10 photon-response
  variants remain the defaults. Every leaf gets a reproducible fresh noise seed;
  CR and CL use adjacent seeds. Keep camera/kernel seeds fixed when you want
  those detector properties shared across acquisitions.

Parameters assigned to the wrong stage are rejected. Shape, recipe, illumination
and beamstop design come from the parent configuration. Unsupported stage changes
should use a separate run rather than trying to reuse an invalid upstream result.
Lengths are metres; photon energies are eV. Acquisition dictionaries and response
callbacks follow [the existing detector model](detector_response.md).

The sample grid is planned at the sampled parent's energy and base detector
geometry, then held fixed across its branches. Native domain/wall resolution
checks still apply. Moving the detector does **not** resize a texture or change
its physical pixel pitch. Choose the base detector geometry/oversampling to cover
the finest resolution required by any branch, and ensure changed reference holes
fit within the fixed sample field. Detector projections can sample only the
reciprocal-space extent represented by that grid; values outside it are zero.

## HDF5 provenance and analysis

Final acquisition pairs remain `00000`, `00001`, etc., with `CR/ideal`,
`CR/detected`, `CR/exit_wave` and corresponding CL arrays. Existing loaders and
[magnetic-pattern analysis](magnetic_pattern_analysis.md) apply unchanged.
Each leaf retains the generation parameters,
domain counts, symmetry measurements, actual detector settings and noise timings.

Under each `metadata/hierarchy/`:

| Dataset | Meaning |
| --- | --- |
| `pattern_id` | Integer independent magnetic parent |
| `exit_wave_id` | `pattern:exit_variant` identifier shared across geometries |
| `ideal_hologram_id` | `pattern:exit_variant:geometry_variant` identifier shared across acquisitions |
| `acquisition_index` | Position in the acquisition list |
| `exit_variant_json`, `geometry_variant_json`, `acquisition_variant_json` | Requested overrides for this leaf |
| `reused_exit_wave`, `reused_ideal_hologram` | Whether that upstream result was cached for this leaf |

Root attributes `pipeline_type` and `hierarchy_variants_json` identify the runner
and complete branching plan. Actual evaluated parameters remain in the standard
metadata groups; callbacks in JSON are descriptive strings. Recorded generation
JSON describes the shared texture's original generation, including its original
timings. Per-leaf pipeline timings measure the work done for that leaf, so cached
propagation/projection stages take nearly zero time. Prefix preparation timings
appear only on the first leaf of each exit branch.

```python
import h5py
from fomocid import DATA_ROOT
with h5py.File(DATA_ROOT / 'Data' / 'hologram_tree' / 'simulation_tree.h5', 'r') as f:
    group = f['00000']
    parent = group['metadata/hierarchy/pattern_id'][()]
    ideal_id = group['metadata/hierarchy/ideal_hologram_id'].asstr()[()]
    exposure = group['metadata/measurement_config/exposure_time'][()]
    detected = group['CR/detected'][()]
```

For ML train/validation/test splits, group by `pattern_id` before splitting.
Randomly splitting leaves would place the same magnetic texture, and potentially
the same ideal hologram, on both sides of the split. More acquisitions increase
measurement diversity without increasing the number of independent magnetic states.

## Validation

The branching tests check actual call counts for pattern generation, sample
propagation and ideal projection; exact equality of cached ideal/exit arrays;
changes with exposure and camera distance; fixed sample pitch across energies;
OH-change rejection; and agreement of the first branch with the independent
pipeline. A complete example with one parent generated all 12 CR/CL pairs.

The current scientific Python environment passes 389 tests plus four subtests
when running `pytest tests --ignore=tests/test_smoke.py`. That full run also
reports three previously established failures: the near-zero diagonal-eigenmode
scalar/Jones comparison and the two repository-wide module/function docstring
contracts. The excluded ML smoke test requires PyTorch, which is unavailable in
that environment. These are verification limits rather than passing checks.

Binary precursors and OH analysis masks are used internally for classification
but are not saved in HDF5. Saved state diagnostics and the wall-smoothed optical field remain available.

### Occasional saturated parents

Both sweep scripts expose `saturation_probability=0.1`. For each independent
magnetic parent, the shared sampler selects an exactly uniform state with 10%
probability, choosing `target_magnetization=-1` or `+1` with equal probability.
Other parents follow the selected free or constrained magnetization mode. An
explicit numeric `target_magnetization` takes precedence over this sampling.
Set the probability to zero to disable it or one to generate only saturation.
This is random sampling, so a short sweep can contain no saturated parents.
In the hierarchical sweep, the draw happens once per magnetic parent; its
energy, mask, geometry and acquisition branches all inherit that same state.
The saved generation parameters record the realized target; the measured
classification is `saturated` with `bubble_count=0`, `stripe_count=1`.

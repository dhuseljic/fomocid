# Reusing simulation stages

Run the new example from the repository root with the project environment active:

```bash
PYTHONPATH=src python tutorials/simulate_hologram_tree.py --patterns 5
```

By default results are saved to `DATA_ROOT / "Data" / "hologram_tree" /
"simulation_tree.h5"`, following the sweep data-folder convention. Override
this with `--output /path/to/file.h5` if needed.

The example creates **30 CR/CL acquisition pairs**: five magnetic patterns × two
exit-wave variants × one detector geometry per parent × three acquisitions. Edit the
`exits`, `geometries` and `acquisitions` lists in that script to change branching.
`--patterns` counts independent magnetic parents, not final output groups.
The example samples distance conditionally between 4 cm and the feasible
upper bound (capped at 40 cm) once per independent magnetic parent. All exit-wave
and acquisition branches under that parent retain its final compatible distance and
real-space grid; distance is not varied in a sub-loop. Each new parent gets a
new draw. At 778 eV and 20 um detector pitch, this range gives at least about
3.18 um of real-space field width, containing the 600 nm OH and reference holes
at 1 um offsets. The earlier 5 mm setting clipped the mask; oversampling alone
cannot enlarge the field. The independent sweep likewise samples its distance
once per experiment from the same domain-dependent interval within 4–40 cm. See
[compatible distances](#compatible-distances-for-fine-magnetic-patterns).

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

Parameters assigned to the wrong stage are rejected. Shape, illumination and beamstop design come from the parent configuration.
The recipe is sampled once through `ranges.recipe` for each independent parent. Unsupported stage changes
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

Both sweep scripts draw one magnetization mode per independent parent:

- **48% free:** `target_magnetization=None`.
- **48% constrained:** a Gaussian target with mean 0 and FWHM 0.5
  (standard deviation about 0.2123), conditioned on [−0.9,+0.9].
- **4% saturated:** exactly −1 or +1, with equal polarity probability.

These are random sampling probabilities, not exact quotas for a short run.
They replace the earlier every-20th-parent saturation schedule. Tree branches
inherit the parent's target, so the mixture applies to holograms as well when
branch counts are equal. Rejection and timeouts can bias the saved proportions.
The script probability settings must sum to one. Set one probability to one
and the others to zero to select a single mode. Realized targets and measured
magnetization remain in the usual HDF5 metadata. The single-experiment GUI
continues to use its explicitly selected target.




### Rejecting oversized parameter draws

Both sweep runners check the required magnetic grid before sample allocation.
A grid above `magnetic_pattern_max_grid_axis` rejects that parameter draw;
the entire experiment (or tree parent) is sampled again. No coarse rescaling
is used. The constructor option `max_sampling_attempts` defaults to 20. If all
attempts fail, that requested parent is skipped and the run continues, keeping
previously saved results. Other errors still stop the run rather than being
silently ignored. A fixed impossible configuration therefore produces no data
for that parent; adjust its physical settings or allocation limit if required.

Root HDF5 attributes `sampling_completed_parents`, `sampling_rejected_draws`,
`sampling_skipped_parents`, `sampling_max_attempts`, and `acquisition_pair_count`
record the actual result. Each saved leaf has `metadata/sampling_attempts`.
Skipped independent-sweep experiments leave gaps in numeric sample IDs; enumerate
existing numeric groups when loading files. Tree parent IDs can likewise have
gaps, while leaf IDs remain consecutive. Rejected proposals bias the accepted
sample distribution: distance is sampled conditionally from 4 cm to the feasible upper limit
(capped at 40 cm), and remaining impossible draws can still bias saved results. In a tree,
no leaves are generated until their parent's grid passes this check.

### Random material recipes

Both examples draw a new recipe once per independent experiment or tree parent:

```text
[Au(t_Au)Cr(t_Cr)]xN/SiN(t_SiN)/Co(t_Co)X(t_X)Y(t_Y)
```

All thicknesses inside recipe strings are **nanometres**, formatted to one
decimal place (for example `24.9`). Cr is quantized to 0.1 nm and Au is four
times that value, preserving the 4:1 ratio and 700–3000 nm stack bounds. Other
thicknesses are rounded to 0.1 nm. The distributions below describe the
continuous draws before this quantization. The shared
`sample_multilayer_recipe(rng)` sampler uses:

| Parameter | Sampling |
| --- | --- |
| `N` | Discrete uniform integer from 7 through 15, inclusive |
| Total Au/Cr thickness `T` | Uniform from 700 to 3000 nm |
| `t_Cr`, `t_Au` | `T/(5*N)` and `4*T/(5*N)`, respectively: exactly 4:1 |
| `t_SiN` | Uniform from 50 to 500 nm |
| `t_Co`, `t_X`, `t_Y` | Independent uniform draws from 2 to 30 nm |
| `X`, `Y` | Two distinct elements drawn without replacement from Pt, Al, Ta, Ir, Cu |

Adjacent terms without `/` are merged by the recipe parser. Each AuCr repeat
is one thickness-weighted effective-medium layer, and CoXY is one merged
magnetic layer. The stack therefore has `N+2` simulated layers (9–17), including
SiN. This reduces per-layer work; it approximates the constituents as mixed
layers rather than resolving separate interfaces. Optical constants are loaded
at the actual branch energy, and the magnetic contribution follows the Co
fraction in the merged CoXY layer.
In the tree, all energies, reference-mask designs and acquisitions under a
parent share its recipe; a new parent gets a fresh recipe. Rejected grid draws
also resample the recipe with the other parent parameters.

The actual recipe is saved in each leaf at `metadata/sample/recipe`; existing
sample metadata also retains expanded layer materials and thicknesses. The
recipe string uses nm while physical layer-thickness metadata uses metres.
Fixed `config.recipe` remains the fallback: set `ranges.recipe=None` to disable
recipe variation, or provide a fixed string or callable override. In `simulate_hologram_sweep.py`, set `randomize_recipe=False` and edit `recipe`
to use the same stack for every experiment. In `simulate_hologram_tree.py`,
pass `--fixed-recipe` to use `config.recipe` for every parent (or set
`randomize_recipe=False` in the script). Random recipes remain enabled by default.
The sampler uses the example's seeded NumPy generator for reproducibility.

The shared sweep sampler keeps weak directional alignment while covering both
ordered and disordered preparations:

- Half of alignment draws are exactly **zero**, removing the preferred-axis penalty.
- The other half use an exponential of scale **0.5**, truncated to [0,24].
  Overall mean alignment is about 0.25; approximately 93% of draws are below 1.
- Quenched spatial `disorder` is uniform from **0 to 0.8**, restoring low-pinning patterns
  that can form smooth labyrinths as well as more disordered patterns.
- `initial_order` is uniform from **0 to 1**, retaining the full range of
  random-to-ordered starting preparations.

Directions remain uniform in [0,π). Saturation scheduling, target magnetization,
wall widths and other physical controls are unchanged. These are input statistics,
not guarantees of measured disorder after relaxation. Grid-limit rejection can
change their distribution among saved experiments. The GUI retains its explicit
single-experiment controls; these distributions apply to the dataset samplers.


The optional constructor setting `experiment_timeout_seconds` also works for a
tree. Its deadline applies to an entire independent parent and all that parent's
branches. A timed-out parent's temporary leaves are all discarded, and the next
parent starts without them. The tree example leaves this setting disabled by
default, since a parent can intentionally contain many acquisitions. See the
[sweep timeout description](simulate_hologram_sweep.md#experiment-timeout) for
platform support and temporary-file behavior.

Sweep output defaults to major-step timings and total elapsed time
(`verbose=True, detailed_logging=False`).
Set `detailed_logging=True` to restore generator, detector and stage breakdowns.
All timing diagnostics are still saved in HDF5. The desktop GUI enables detailed
logging in its dedicated progress pane.

See [Choosing the HDF5 save mode](hdf5_save_modes.md) for script, command-line
and GUI controls, storage precision, lossless roundtrip checks and readback.

The shared magnetic sampler draws physical repeat lengths log-uniformly from
64–1000 nm. Half of preparations use the smooth, isotropic Notebook 22-style
relaxation; the remainder retain varied preparation order and disorder. See the
[full distributions](constrained_magnetic_domains.md) and the
[physical beamstop diffraction option](single_experiment_gui.md#physical-beamstop-diffraction-optional),
which remains disabled by default. Stored morphology uses full periodic geometry
for bubble/stripe decisions and open OH intersections for stripe counts (v7).

### Compatible distances for fine magnetic patterns

Both examples enable `magnetic_pattern_distance_min=0.04` (metres) and use
`ranges.detector_distance=0.40` as the upper bound. First draw the physical
magnetic spacing, wall width and other magnetic controls. Then sample distance
**uniformly from 4 cm to the smaller of 40 cm and the compatible upper limit**.
This directly samples the feasible interval, rather than first drawing an
independent distance and adjusting only failures. Magnetic controls stay intact.

Compatibility uses the actual discrete detector q-span, energy, pixel size,
detector center, integer oversampling budget and all constraints in
`constrained_sampling_pitch` (spacing, wall sigma and minority-width estimate).
Sampling runs before aperture/beamstop callbacks, which see the final distance.
A tree chooses it once per parent and shares it across every downstream branch.
The allocation limit still applies: a draw unresolved even at 4 cm is rejected
and the existing whole-parent retry/skip policy continues. This can leave residual
selection bias; grids are never silently coarsened.

Final distance and magnetic resolution are correlated. The final distance is
`metadata/detector/sample_to_detector_distance`. Under `metadata/detector/`,
`distance_sampling_upper_m` records the input upper bound (40 cm in the examples),
`compatible_distance_min_m` records the lower bound, and
`distance_adjusted_for_magnetic_grid` compares the final distance with that
input upper bound. The legacy `sampled_distance_m` field also records the input
upper bound; it is not a separate initial random draw in this mode.
Set `magnetic_pattern_distance_min=None` to disable conditional sampling and
interpret detector_distance as the actual distance; manual GUI experiments use
that default.

### Energy distribution around the Co absorption peak

Both examples use `sample_absorption_edge_energy(rng)`: a Gaussian centered on
the maximum of the bundled Co L-edge absorption table (`Co_beta.txt`), with
underlying sigma **1 eV**, truncated to **peak +/-2 eV** by rejection sampling.
The present table maximum is **781.13866 eV**, giving bounds **779.13866–783.13866 eV**.
This uses total absorption beta, not the dichroism maximum or nominal edge onset.
No out-of-range draw is clipped to an endpoint. Truncation makes the resulting
standard deviation about 0.880 eV despite the underlying Gaussian sigma of 1 eV.

Sweep energy is drawn once per experiment. Tree energy is drawn once per parent
and shared across its mask/acquisition branches; the example's fixed 780 eV
branch override has been removed. The library still supports explicit energy
branches. The actual simulated energy remains available in each experiment's
`metadata/xray/energy_eV` dataset. Distance compatibility is evaluated after energy
sampling, so its wavelength is included in the sampling check.

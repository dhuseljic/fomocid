# Magnetic patterns: a collaborator's analysis guide

This guide describes files produced by `tutorials/simulate_hologram_sweep.py`.
You can read them with **Python, NumPy and h5py**, without running the simulator.
Matplotlib is only needed for the plotting example. Use the
[generator reference](constrained_magnetic_domains.md#production-hologram-generation)
for the complete generation-control table and model equations.

## What one sample contains

The file has numeric sample groups such as `00000`, `00001`, …, and a
`_pipeline_config` group. One numeric group is one generated specimen/experiment,
with paired `CR/` and `CL/` holograms. Those two helicities use the same specimen;
they are not separate magnetic-pattern samples. A failed/interrupted run may
have an incomplete final group; check that the required datasets exist.

The magnetic field is a 2-D normalized out-of-plane magnetization `mz = Mz/Ms`,
not magnetization in A/m. Both polarities are physical magnetic domains. The
same lateral field is assigned to the magnetic material layers; the stored
2-D map is not a resolved 3-D vector micromagnetic simulation.

Use these three representations for different analyses:

| Representation | Where, relative to a numeric sample group | Meaning |
|---|---|---|
| Binary precursor | `metadata/sample/magnetic_pattern/binary_pattern` | ±1 array on the **generation grid**, before wall blur. Use for topology, connected domains and classification. |
| OH selection | `metadata/sample/magnetic_pattern/analysis_mask` | Same shape as the binary precursor; true means visible through the object hole. Use to select visible domains/pixels. |
| Optical pattern | `magnetic_pattern_oh` | Smoothed `mz` multiplied by the OH mask, on a possibly different saved crop. Zero outside OH means masked data, **not** a physical zero-magnetization domain. |

Gaussian blur uses physical `sigma` as a standard deviation, not FWHM. Wall
stiffness controls the relaxation model and is a different parameter. The
optical wall convention is `(mx,my,mz)=(0,sqrt(1-mz²),mz)`. It preserves unit
magnitude through the wall, but is not a vector micromagnetic or chirality model.

Detector interpretation is separate from magnetic classification. The default
transfer curve is linear (`metadata/detector_response/model = "linear"`), with
noise and ADC clipping still enabled. `CR/ideal` and `CL/ideal` are ideal photon
rates, while detected arrays are acquired ADU averaged over frames. Use
`metadata/detector_response/{CR,CL}/counts_per_photon` for evaluated gain;
`detector_params/nonlinear_response` records any response callback identifier.
See the [detector-response guide](detector_response.md#paired-cr-cl-hologram-sweep)
for units, acquisition mode and nonlinear options. Older files may lack these
new explicit response records.

## Inputs versus achieved properties

There are two independent options for controlling targets:

- `target_magnetization=None`: no sign-area constraint. The mean magnetization
  and up fraction emerge from initialization and relaxation and are measured.
- Numeric `target_magnetization` in [-1,1]: constrain the binary sign area during
  every relaxation step. The full-generation up-pixel count is the nearest
  integer to `N * (1 + target_magnetization)/2`. This does not force the same
  fraction inside the OH. ±1 produces a homogeneous field.
- `max_length_iterations=1`: one relaxed realization, **no spacing feedback**.
  `characteristic_length` is the preferred model spacing, not a guaranteed
  measured period, bubble diameter or stripe width.
- `max_length_iterations>1`: optional same-seed retries to calibrate the measured
  spacing. Requested and achieved values and all attempts are recorded.

The sweep defaults to free magnetization and one relaxation run. Near its
magnetic-pattern controls, set `target_magnetization=-0.6` to use a fixed target,
or set `constrain_magnetization=True` with `target_magnetization=None` to sample
a numeric target for each specimen. Free evolution need not produce a balanced
mixture of state labels and can reach saturation naturally.

| Metadata | Location under `metadata/sample/magnetic_pattern/` | How to interpret it |
|---|---|---|
| Physical inputs | `pattern_config/` | Metres, radians and dimensionless controls actually passed through the pipeline. HDF5 stores an optional input `None` as the text `"None"`. |
| Generator provenance | `generation_json` | Effective controls, model/solver name, seed, pixel-unit scales, relaxation trace, achieved full-grid properties and optional calibration history. JSON null represents `None`. |
| Production state | `classification_json`; also `state`, `bubble_count`, `stripe_count` | Authoritative production classification using periodic geometry and OH visibility. Includes per-component records. |
| Production measurements | `measurements_json`; also `measurements/` | Full-grid size/occupancy and explicitly named OH measurements, plus OH-selected counts/order. JSON undefined values are null; numeric datasets use NaN. |
| Method/convention | `classification_method`, `classification_periodic`, `classification_geometry` | Read these for the actual stored algorithm and boundary convention; older files may use another convention. |

`generation_json["measurements"]` contains the low-level generator's diagnostic
measurements, which can use the notebook's open-field convention and the full
box. Production normally sets `generation_json["component_diagnostics"]=false`
and omits redundant full-box component counts there; spectral length and
occupancy are still present. For comparisons with the hologram, use **production** `classification_json`
and `measurements_json`; do not substitute the nested generator measurements.

## Counts, state labels and visibility

Production constrained patterns have periodic boundaries on their **generation
grid** (the generated ROI when optimization is enabled). Opposite-edge pieces
are joined and unwrapped before measuring their full shape. Only components
with at least one pixel in the OH contribute. A component visible in two pieces
or on opposite edges counts once; entirely hidden domains are excluded.
Counts/order are computed on the binary precursor, not on wall-blurred data.

Default classification settings are full area >=9 pixels, eccentricity <=0.95
(approximately a 3.2:1 ellipse), circularity >=0.4, threshold zero and eight-neighbor
connectivity. Small components are noise; other resolved non-bubble domains are
stripe-like. The connected majority background also counts as a stripe-like
domain. A connected labyrinth is one component, not a count of its branches.

Total counts add both polarities. `positive_*`, `negative_*`, `majority_*` and
`minority_*` retain separate bubble, stripe and noise counts. Majority/minority
is defined over the full generation grid, not the OH; positive is called
minority at equal fractions. Use polarity plus label to identify a component:
label numbers alone are only unique within one polarity and one sample.

| Total bubbles B and stripe-like domains S | `state` |
|---|---|
| B>0, S=1 | `bubble`: bubbles on one connected background |
| B=0, S>1 | `stripe` |
| B>0, S>1 | `mixed` |
| B=0, S=1 | `saturated` |
| B>0, S=0 | `bubble`: the OH excludes the background |
| B=0, S=0 | `unresolved`: nothing passes the area cutoff |

The raw `morphology` field describes the component mixture; it can say `mixed`
for bubbles plus one background while **`state` correctly says `bubble`**. Use
`state` for the table above. A saturated label can coexist with discarded small
islands; `noise_count` and `is_uniform` distinguish this from exact full-box
uniformity. Counts depend on resolution, thresholds and OH size.

`boundary_fragment_count` is retained for open-field analysis compatibility;
it is zero for complete periodic production classification. Open Notebook 22
gallery counts can differ from periodic production counts at the box edges.

## Size, order and component diagnostics

| Field | Meaning / units |
|---|---|
| `measurements.up_fraction`, `mean_mz` | Binary occupancy and normalized mean over the **full generated box**. `mean_mz = 2*up_fraction - 1`. |
| `measurements.up_fraction_in_oh`, `mean_mz_in_oh` | Same quantities using OH pixels only; these can differ from the requested target. |
| `measurements.dominant_period_px` | Inverse frequency of the largest individual 2-D Fourier-power pixel, computed on the full binary box. Multiply by the sample pixel pitch for metres. |
| `generation.measured_length_px` | Interpolated radial Fourier-power peak period on the full generated box; this is the length used by optional calibration. It need not match the single-pixel dominant period. |
| `generation.length_relative_error`, `length_target_met` | Comparison with the preferred/requested spacing; diagnostic only when calibration is disabled. Uniform states have no defined period. |
| `generation.wall_sigma_m`, `wall_sigma_px` | Applied Gaussian wall sigma in metres and native pixels. |
| `generation.mean_mz_after_wall_smoothing`, `up_fraction_after_wall_smoothing` | Full-box properties after blur. These do not describe the masked optical array including its external zeros. |
| `measurements.mean_eccentricity`, `mean_circularity` | Means over resolved, OH-visible **minority** components; not averages over both phases. |
| `stripe_nematic_order` | 0–1 alignment statistic of resolved stripe-like component major axes, area/anisotropy weighted, both polarities. One anisotropic background can give 1; this is not local stripe periodicity. |
| `local_sixfold_order` | Mean magnitude of six-nearest-bubble bond order, OH-selected centers with periodic distances. Requires at least seven bubbles. |
| `sixfold_order` | Magnitude after complex averaging, indicating a common hexagonal bond orientation. Differently oriented grains can have high local but lower global order. |
| `interior_bubble_count` / `measurements.interior_component_count` | Number of bubble centers used for bond-order measurement, not an additional count to add to B. |

Per-component entries in `classification_json["components"]` contain:

- `polarity` (+1/-1), `label`, `is_bubble` and `is_boundary_fragment`.
- `area`: full periodic component area in pixels, including pixels outside OH.
- `visible_area`: overlap with OH in pixels. Multiply either area by `pitch**2`
  for m², or by `(pitch*1e9)**2` for nm². `min_area` applies to full `area`.
- `eccentricity`: covariance-based shape elongation; zero is isotropic.
- `circularity`: `4*pi*area/perimeter**2`, using a digital pixel-edge perimeter;
  pixelated circles do not necessarily have value one.
- `touches_aperture`: some of the component lies outside OH.
- `touches_border`: it intersects the generation-array edge. With periodic
  geometry this does **not** imply incomplete shape or reject a bubble.

These descriptors measure shape and orientational order, not skyrmion charge,
crystallographic symmetry or physical equilibrium. `converged`, `residual`,
`steps_completed`, `trace`, `calibration_enabled` and `calibration_history` in
`generation_json` describe solver progress. Convergence and matching a spacing
are different checks. With free magnetization, `target_up_pixels` is null and
`magnetization_constrained` is false; achieved values are always measured.

## Load one specimen and its physical coordinates

The following loader reads all arrays before closing the file. It targets the
current sweep layout and deliberately reports missing fields instead of guessing
values for older files. Length inputs in `pattern_config` are metres; effective
lengths in generator metadata are generally pixels unless named `_m`.

```python
import ast
import json
from pathlib import Path

import h5py
import numpy as np


def read_scalar(dataset):
    value = dataset[()]
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    elif isinstance(value, np.generic):
        value = value.item()
    return None if isinstance(value, str) and value == "None" else value


def load_magnetic_sample(path, sample_id="00000"):
    with h5py.File(path, "r") as h5:
        g = h5[sample_id]
        meta = g["metadata/sample/magnetic_pattern"]
        required = ("binary_pattern", "analysis_mask", "generation_json",
                    "classification_json", "measurements_json")
        missing = [key for key in required if key not in meta]
        if missing:
            raise ValueError(f"{sample_id}: missing magnetic datasets: {missing}")
        binary = meta["binary_pattern"][()]
        oh = meta["analysis_mask"][()].astype(bool)
        pitch = float(g["metadata/sample/real_space_pixel_size"][()])
        factor = int(g["metadata/sample/effective_oversampling"][()])
        detector_shape = read_scalar(g["metadata/detector/shape"])
        if isinstance(detector_shape, str):
            detector_shape = ast.literal_eval(detector_shape)
        full_shape = np.array(detector_shape, dtype=int) * factor

        def extent(shape, prefix):
            starts = [int(meta[f"{prefix}{axis}_start_px"][()])
                      if f"{prefix}{axis}_start_px" in meta else 0
                      for axis in ("y", "x")]
            y0, x0 = (np.array(starts) - full_shape/2 - .5) * pitch * 1e9
            ny, nx = shape
            return (x0, x0 + nx*pitch*1e9, y0, y0 + ny*pitch*1e9)

        optical = g["magnetic_pattern_oh"][()]
        return dict(
            sample_id=sample_id, binary=binary, oh=oh, optical=optical,
            pitch_m=pitch, binary_extent_nm=extent(binary.shape, "roi_"),
            optical_extent_nm=extent(optical.shape, "saved_roi_"),
            parameters={key: read_scalar(ds) for key, ds in meta["pattern_config"].items()},
            generation=json.loads(meta["generation_json"].asstr()[()]),
            classification=json.loads(meta["classification_json"].asstr()[()]),
            measurements=json.loads(meta["measurements_json"].asstr()[()]),
            method=meta["classification_method"].asstr()[()],
            cr_ideal=g["CR/ideal"][()], cl_ideal=g["CL/ideal"][()],
        )
```

Set the file path, list available specimens and inspect one:

```python
path = Path("simulation_sweep.h5")  # replace with your file path
with h5py.File(path, "r") as h5:
    sample_ids = sorted(key for key in h5 if key.isdigit())
print("Specimens:", sample_ids)

sample = load_magnetic_sample(path, sample_ids[0])
print("Physical inputs:", sample["parameters"])
print("Target constrained:", sample["generation"]["magnetization_constrained"])
print("Measured state:", sample["classification"]["state"])
print("B, S:", sample["classification"]["bubble_count"],
      sample["classification"]["stripe_count"])
print("OH mean mz:", sample["measurements"]["mean_mz_in_oh"])
print("Pitch (nm):", sample["pitch_m"] * 1e9)
```

## Plot topology and the field used by the optics

These arrays use `(row=y, column=x)`. The extents below are in nanometres relative
to the full simulation center, using the simulator's index-minus-half-size
convention. The binary generation ROI and saved optical ROI can have different
shapes/origins. Do not directly multiply or index them against each other.
`supportmask` is on the detector-derived reconstruction grid and is not a
replacement for the generation-grid `analysis_mask`.

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
images = [sample["binary"], np.ma.masked_where(~sample["oh"], sample["binary"]),
          sample["optical"]]
labels = ["Full generated binary field", "Binary field inside OH", "Smoothed mz × OH"]
extents = [sample["binary_extent_nm"]] * 2 + [sample["optical_extent_nm"]]
for ax, image, label, extent in zip(axes, images, labels, extents):
    ax.imshow(image, cmap="RdBu_r", vmin=-1, vmax=1, origin="lower", extent=extent)
    ax.set(title=label, xlabel="x (nm)", ylabel="y (nm)", facecolor="0.85")
plt.show()
```

The generation box is deliberately compact around the actual OH, with native
fine sampling, a configurable context margin and FFT-friendly dimensions.
`generation_shape_px`, `context_periods`, ROI insertion bounds and
`effective_oversampling` record this choice. Changing box size changes the
realized texture for a seed; seeds alone do not define an identical specimen
across different grids or code revisions.

## Export one row per specimen for analysis

This example uses the loader above and Python's standard CSV library, with no
pandas dependency. It keeps requested inputs and achieved quantities in separate
columns. Null order/period values become empty CSV cells.

```python
import csv

rows = []
for sample_id in sample_ids:
    s = load_magnetic_sample(path, sample_id)
    c, m, gen, inputs = (s[key] for key in
                         ("classification", "measurements", "generation", "parameters"))
    period_px = m["dominant_period_px"]
    rows.append(dict(
        sample=sample_id, seed=gen["seed"], method=s["method"],
        magnetization_constrained=gen["magnetization_constrained"],
        requested_mz=inputs.get("target_magnetization"),
        achieved_mz_full=m["mean_mz"], achieved_mz_oh=m["mean_mz_in_oh"],
        state=c["state"], bubbles=c["bubble_count"], stripes=c["stripe_count"],
        positive_bubbles=c["positive_bubble_count"], negative_bubbles=c["negative_bubble_count"],
        noise=c["noise_count"], pitch_nm=s["pitch_m"]*1e9,
        preferred_spacing_nm=inputs["characteristic_length"]*1e9,
        measured_dominant_spacing_nm=None if period_px is None else period_px*s["pitch_m"]*1e9,
        wall_sigma_nm=gen["wall_sigma_m"]*1e9,
        stripe_order=c["stripe_nematic_order"],
        local_hexagonal_order=c["local_sixfold_order"],
        global_hexagonal_order=c["sixfold_order"],
        converged=gen["converged"], calibration_enabled=gen["calibration_enabled"],
    ))
if rows:
    csv_path = path.with_name("magnetic_summary.csv")
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("Saved:", csv_path)
```

For large files, omit the hologram reads from `load_magnetic_sample` when making
metadata-only tables. Holograms can be substantially larger than the diagnostics.

## Inspect where generation time was spent

```python
with h5py.File(path, "r") as h5:
    stages = h5[f"{sample['sample_id']}/metadata/timing_seconds"]
    print("Pipeline seconds:", {key: float(value[()]) for key, value in stages.items()})
print("Generator seconds:", sample["generation"]["timings_seconds"])
```

Generator `relaxation` contains the local-update, FFT, area/clipping,
convergence and trace substeps. `total` includes all generator attempts but not
wall smoothing. These nested values must not be added twice. See
[timing and progress definitions](constrained_magnetic_domains.md#timing-and-live-generation-progress)
for all fields and the optional progress callback. Older files may lack timings.

## Reproduce the stored component analysis

If the simulator source is installed/on `PYTHONPATH`, the stored settings can
rebuild component-label images. Such images are returned by the classifier but
are not themselves saved in HDF5. Both label images use the generation grid and
retain full geometry for OH-visible components.

```python
from scattering_calculator.sample_generator.domain_analysis import classify_magnetic_domains

c = sample["classification"]
recomputed = classify_magnetic_domains(
    sample["binary"], analysis_mask=sample["oh"],
    threshold=c["classification_threshold"], min_area=c["classification_min_area_px"],
    max_eccentricity=c["bubble_max_eccentricity"],
    min_circularity=c["bubble_min_circularity"], connectivity=c["classification_connectivity"],
    periodic=c["periodic"], boundary_context_periodic=c.get("boundary_context_periodic", False),
)
assert (recomputed["bubble_count"], recomputed["stripe_count"]) == (c["bubble_count"], c["stripe_count"])
positive_labels = recomputed["positive_label_image"]
negative_labels = recomputed["negative_label_image"]
```

Keep the original HDF5 alongside exported tables: it contains the precursor,
selection mask, realized parameters and diagnostic history. To reproduce the
**generation**, also share the source revision and numerical environment; a full
source/environment snapshot is not embedded in this file. Existing files are
not relabelled automatically when classification code changes.

The separate `simulate_experiment` backend uses
`runs/000000/magnetic_state/` instead of numeric top-level specimen groups. See
[the backend layout note](constrained_magnetic_domains.md#diagnostics-and-hdf5-provenance)
for its `parameters_json`, `generation_json`, `binary_pattern`, optional OH
`analysis_mask`, and named `classification/` datasets. This guide's loader is
specifically for paired-CR/CL sweep files.

# Simulate Hologram Sweep

`tutorials/simulate_hologram_sweep.py` generates a sweep of simulated FTH
holograms and saves the result as one HDF5 file. Each simulated sample gets a
randomized x-ray energy, magnetic pattern, holography mask, illumination,
detector distance, beamstop, and detector acquisition settings.

The script is useful for producing training or benchmarking datasets where each
HDF5 group is one complete synthetic experiment.

## Running The Sweep

From the repository root:

```bash
python3 tutorials/simulate_hologram_sweep.py
```

The output path is set near the top of the script:

```python
output_folder = DATA_ROOT / "Data" / "hologram_sweep"
output_path = output_folder / "simulation_sweep.h5"
```

The number of simulated configurations is controlled at the bottom:

```python
nr_simulations = 20
```

Set `pipeline_random_seed` to an integer for reproducible sweeps, or to `None`
for non-reproducible random draws:

```python
pipeline_random_seed = 0
```

## What The Script Does

For each sample, the pipeline:

1. Samples the requested random parameters.
2. Builds the x-ray, detector, beamstop, sample, magnetic pattern, aperture, and illumination configs.
3. Generates an OH-local magnetic pattern when ROI mode is enabled.
4. Builds the holography support mask and detector beamstop.
5. Computes the dielectric tensor and Jones propagation for CR and CL polarization.
6. Creates ideal holograms, applies detector effects/noise, and saves everything to HDF5.

Timing for each stage is printed when `verbose=True`.

## Important Controls

### ROI Mode

These flags reduce work by only generating or computing expensive quantities in
regions that matter:

```python
use_roi = True
dielectric_tensor_use_roi = True
```

`use_roi=True` enables OH-local magnetic patterns and local aperture/dielectric
optimizations. Turn it off to compute on the full slab.

### Magnetic Patterns

The fixed fallback pattern is selected with:

```python
pattern_type = "binary_labyrinth_pattern"
```

Supported options:

- `binary_labyrinth_pattern`: labyrinth domains generated from the binary Gray-Scott helper, rescaled to the requested stripe width.
- `disordered_skyrmion_lattice_pattern`: random non-overlapping irregular skyrmions; `stripe_width` is the average skyrmion diameter.
- `saturated_pattern`: uniform `mz = +1` or `mz = -1`.
- `wavy_stripe_pattern`: analytic wavy stripe domains.

The sweep currently chooses pattern types with this approximate mixture:

```python
pattern_type=Choice(
    (
        "binary_labyrinth_pattern",
        "binary_labyrinth_pattern",
        "binary_labyrinth_pattern",
        "disordered_skyrmion_lattice_pattern",
        "disordered_skyrmion_lattice_pattern",
        "disordered_skyrmion_lattice_pattern",
        "saturated_pattern",
    )
)
```

That is 3/7 labyrinth, 3/7 skyrmions, and 1/7 saturated. Add or remove repeated
entries to change the mixture.

### Pattern Size Parameter

`stripe_width` is the common texture-size parameter used by downstream detector
and aperture range logic:

```python
stripe_width = Uniform(10e-9, 500e-9).sample()
```

For labyrinth and stripe patterns, it means stripe/domain width. For skyrmions,
it means average skyrmion diameter. For saturated states, it is still sampled so
the detector and aperture geometry can be chosen consistently, but the magnetic
pattern itself ignores it.

### Skyrmion Parameters

The random skyrmion generator uses:

- `skyrmion_density`: approximate target skyrmion area fraction.
- `diameter_spread`: bounded half-range for diameter variation, in metres in the config.
- `ellipticity`: y/x axis-ratio range.
- `roughness`: boundary irregularity amplitude.
- `roughness_modes`: Fourier modes used for the rough boundary.

Each skyrmion gets its own random diameter, ellipticity, rotation angle, and
rough boundary. Candidate centers are sampled randomly inside the OH placement
region, using the OH radius plus one average skyrmion diameter as the placement
radius. A center candidate is tried first so low-density skyrmion samples still
contain at least one skyrmion inside the OH field of view. Candidates are
accepted only if they do not overlap previously accepted skyrmions.

### X-ray Ranges

The sweep randomizes the photon energy across the Co L-edge range and also
randomizes the transverse coherence length:

```python
xray_energy=Uniform(775, 795)
xray_coherence_length=random_coherence_length
```

`random_coherence_length` first samples a shared base coherence length from
`5e-6` to `50e-6` metres, then applies a small opposite y/x anisotropy:

```python
def random_coherence_length(params):
    base = Uniform(5e-6, 50e-6).sample()
    anisotropy = Uniform(-0.13, 0.13).sample()
    return (base * (1.0 + anisotropy), base * (1.0 - anisotropy))
```

This keeps the two components similar while allowing up to about a 30% ratio
difference between axes. The tuple order is `(y, x)`.

### Illumination Ranges

The sweep randomizes the Gaussian illumination:

```python
illumination_focus_distance=Uniform(0.0, 2e-3)
illumination_fwhm=Uniform(5e-6, 50e-6)
illumination_center=(
    Uniform(-2e-6, 2e-6),
    Uniform(-2e-6, 2e-6),
)
```

The center tuple is `(y, x)` in metres.

### Measurement Ranges

The sweep randomizes detector acquisition settings:

```python
measurement_config=lambda params: {
    "number_frames": int(np.random.randint(1, 21)),
    "max_counts_per_image": Uniform(40_000, 70_000).sample(),
    "exposure_time": measurement_config["exposure_time"],
}
```

`exposure_time` is currently fixed.

## HDF5 Layout

The output file contains one top-level group per simulation:

```text
simulation_sweep.h5
├── _pipeline_config/
├── 00000/
├── 00001/
└── ...
```

Each sample group contains:

```text
00000/
├── CR/
│   ├── exit_wave
│   ├── ideal
│   └── detected
├── CL/
│   ├── exit_wave
│   ├── ideal
│   └── detected
├── beamstop_mask
├── supportmask
├── magnetic_pattern_oh
└── metadata/
```

Polarization groups:

- `CR/ideal`, `CL/ideal`: ideal detector holograms before detector noise.
- `CR/detected`, `CL/detected`: detector holograms after noise/artifacts.
- `CR/exit_wave`, `CL/exit_wave`: complex exit wavefields.

Other arrays:

- `beamstop_mask`: detector beamstop mask.
- `supportmask`: binary support mask in FTH reconstruction coordinates.
- `magnetic_pattern_oh`: magnetic pattern inside the OH save ROI; pixels outside the OH are zeroed.

Metadata is stored as nested scalar/list datasets below `metadata/`, including
x-ray, detector, measurement, illumination, aperture, magnetic-pattern, ROI, and
propagation parameters.

## Reading The HDF5 File

Basic inspection:

```python
import h5py

path = "/path/to/simulation_sweep.h5"

with h5py.File(path, "r") as h5:
    sample_keys = [k for k in h5.keys() if k != "_pipeline_config"]
    print(sample_keys[:5])
    print(list(h5[sample_keys[0]].keys()))
```

Read holograms and masks:

```python
import h5py

with h5py.File(path, "r") as h5:
    g = h5["00000"]

    cr_ideal = g["CR/ideal"][()]
    cr_detected = g["CR/detected"][()]
    cl_detected = g["CL/detected"][()]
    exit_wave = g["CR/exit_wave"][()]

    beamstop = g["beamstop_mask"][()]
    support = g["supportmask"][()]
    magnetic_oh = g["magnetic_pattern_oh"][()]
```

Read metadata:

```python
with h5py.File(path, "r") as h5:
    m = h5["00000/metadata"]

    energy = m["xray/energy"][()]
    pattern_type = m["magnetic_pattern/pattern_type_method"][()].decode()
    detector_distance = m["detector/detector_distance"][()]
    illumination_center = m["illumination/center_m"][()]
```

Some string datasets are stored as bytes, so use `.decode()` when needed.

Recursively flatten metadata into a dictionary:

```python
import h5py

def read_h5_group(group, prefix=""):
    out = {}
    for key, item in group.items():
        path = f"{prefix}/{key}" if prefix else key
        if isinstance(item, h5py.Dataset):
            value = item[()]
            if isinstance(value, bytes):
                value = value.decode()
            out[path] = value
        else:
            out.update(read_h5_group(item, path))
    return out

with h5py.File(path, "r") as h5:
    meta = read_h5_group(h5["00000/metadata"])

print(meta["xray/energy"])
print(meta["magnetic_pattern/pattern_type_method"])
```

## FTH Reconstruction

The pipeline stores holograms, not a separate reconstruction array. A simple
FTH-style reconstruction from an ideal or detected hologram is:

```python
import numpy as np

def fth_reconstruct(hologram):
    return np.fft.fftshift(
        np.fft.fft2(np.fft.fftshift(hologram, axes=(-2, -1)), axes=(-2, -1)),
        axes=(-2, -1),
    )

with h5py.File(path, "r") as h5:
    holo = h5["00000/CR/detected"][()]
    recon = fth_reconstruct(holo)
```

Use `np.abs(recon)` for amplitude or `np.angle(recon)` for phase.

## Common Edits

Change output size:

```python
nr_simulations = 1000
```

Turn off reproducibility:

```python
pipeline_random_seed = None
```

Force one magnetic pattern type:

```python
ranges = HologramPipelineRanges(
    pattern_type="binary_labyrinth_pattern",
    pattern_config=random_pattern_config,
    ...
)
```

Disable ROI optimizations:

```python
use_roi = False
dielectric_tensor_use_roi = False
```

Change detector image size:

```python
detector_pixel_shape = (1300, 1300)
```

## Notes

- Physical lengths in pattern configs are specified in metres. The pipeline
  converts them to pixels using the simulation real-space pixel size.
- `magnetic_pattern_oh` may be saved as an OH ROI rather than the full simulated
  slab when ROI mode is enabled. Its ROI offsets are saved under
  `metadata/magnetic_pattern/saved_roi_*`.
- Callable range functions receive the already-sampled parameter dictionary, so
  they can build dependent ranges such as detector distance from energy and
  texture size.

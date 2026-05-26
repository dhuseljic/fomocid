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

The number of simulated configurations is controlled near the top:

```python
nr_simulations = 2
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

### Multislice Propagation

Free-space propagation between material layers is controlled near the top of
the script:

```python
propagate = True
propagation_padding_px = 128
propagation_padding_mode = "edge"
propagation_absorber_width_px = 64
propagation_absorber_strength = 6.0
propagation_absorber_profile = "cosine"
```

When `propagate=True`, the Jones field is propagated between consecutive
material layers with an angular-spectrum free-space propagator using the actual
sample pixel size and each layer thickness. This is more physical for thick
or strongly structured aperture stacks, but it is slower because it adds FFTs
between layers. When `propagate=False`, the simulator applies only the local
Jones transmission for each layer, which is the faster historical mode.

The free-space propagator damps evanescent spatial frequencies to avoid
unphysical exponential growth and caches repeated propagation kernels for
repeated layer thicknesses. Because angular-spectrum propagation is FFT-based,
the sweep exposes several boundary controls:

- `propagation_padding_px`: padding added on each side for every
  free-space step, then cropped away after propagation. Set to `0` to disable.
- `propagation_padding_mode`: padding strategy. `"edge"` and `"reflect"` avoid
  introducing a hard zero wall at the original crop boundary; `"constant"` keeps
  the historical zero padding.
- `propagation_absorber_width_px`: smooth edge-absorber width. If padding is
  enabled, the absorber is clamped to the padded margin so it does not attenuate
  the returned field. Set to `0` to disable.
- `propagation_absorber_strength`: maximum exponential absorber strength.
  Larger values damp boundary wraparound more aggressively.
- `propagation_absorber_profile`: how absorption ramps from the interior to the
  edge. `"cosine"` and `"smoothstep"` are usually smoother than a linear ramp.

The previous per-polarization implementation is kept in
`propagate_free_space_jones_260526` for future comparisons.

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
        "binary_labyrinth_pattern",
        "binary_labyrinth_pattern",
        "disordered_skyrmion_lattice_pattern",
        "disordered_skyrmion_lattice_pattern",
        "disordered_skyrmion_lattice_pattern",
        "saturated_pattern",
    )
)
```

That is 5/9 labyrinth, 3/9 skyrmions, and 1/9 saturated. Add or remove
repeated entries to change the mixture.

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

For `binary_labyrinth_pattern`, `auto_size=True` estimates how much the
generated labyrinth image will be rescaled to reach the requested stripe width.
Large final stripes therefore use a smaller generated source field when
possible, while still regenerating a larger field if the measured FFT stripe
width would make the scaled image too small to crop safely.

### Beamstop And Wire Mask

The beamstop can include a bent support wire:

```python
beamstop_config = {
    "radius": 0.2e-3,
    "sigma": 20e-6,
    "wire_width": 0.05e-3,
    "wire_bend": 0.1e-3,
    "antialias": 4,
}
```

`antialias` supersamples the beamstop and wire mask before averaging back to
detector pixels. Values above `1` reduce stair-step artifacts for thin or
diagonal wires and produce fractional mask values at edges. Use `1` for the
historical binary rasterization.

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

### Aperture Geometry And Roughness

The FTH mask contains one object hole (`OH`) and a random number of reference
holes (`RH`). Apertures are layer-aware:

- `aperture_top_radius_factors` controls the top/base radius ratio.
- The taper is derived from the material stack above the SiN membrane.
- The two layers closest to the SiN membrane remain cylindrical; if the stack
  above SiN is only two layers, the aperture stays cylindrical.
- Deeper layers below the taper keep the base aperture size.

Boundary roughness is specified in physical units in the sweep script:

```python
aperture_roughness_amplitude = 20e-9
aperture_roughness_period = 10e-9
```

`aperture_roughness_from_length` converts those physical values into the
relative roughness amplitude and Fourier-mode band used by the aperture
generator. The helper caps the relative roughness at `0.25` by default. The OH
remains circular (`ellipticity = 1`, `angle = 0`) but has high-frequency
boundary roughness. RHs can still be elliptical/rotated,
with the same physical roughness scale. Rough contours are regenerated per
layer using deterministic per-layer seeds, so a tapered aperture is not just
the same rough outline rescaled through depth.

After the sweep runs, the script rebuilds the first sample's aperture mask from
metadata and plots the depth-averaged mask plus Y-Z/X-Z cuts through the OH/RHs.
These aperture depth diagnostics are not saved as extra HDF5 datasets.

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

- `beamstop_mask`: detector beamstop mask; anti-aliased edges can be fractional.
- `supportmask`: binary support mask in FTH reconstruction coordinates.
- `magnetic_pattern_oh`: magnetic pattern inside the OH save ROI; pixels outside the OH are zeroed.

Metadata is stored as nested scalar/list datasets below `metadata/`, including
x-ray, detector, measurement, illumination, aperture, magnetic-pattern, ROI, and
propagation parameters.

The `_pipeline_config/` group stores fixed top-level settings such as `recipe`,
`oversampling`, detector shape, `propagate`, `propagation_padding_px`,
`propagation_padding_mode`, `propagation_absorber_width_px`,
`propagation_absorber_strength`, and `propagation_absorber_profile`.

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
    aperture_roughness = m["aperture/aperture_config/apertures_roughness"][()]
    propagate = h5["_pipeline_config/propagate"][()]
    propagation_padding_px = h5["_pipeline_config/propagation_padding_px"][()]
    propagation_padding_mode = h5["_pipeline_config/propagation_padding_mode"][()].decode()
    propagation_absorber_width_px = h5["_pipeline_config/propagation_absorber_width_px"][()]
    propagation_absorber_profile = h5["_pipeline_config/propagation_absorber_profile"][()].decode()
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

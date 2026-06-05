# Simulate Hologram Sweep

`tutorials/simulate_hologram_sweep.py` generates a sweep of simulated FTH
holograms and saves the result as one HDF5 file. Each simulated sample gets a
randomized x-ray energy, magnetic pattern, holography mask, illumination,
detector distance, beamstop, and detector acquisition settings.

The script is useful for producing training or benchmarking datasets where each
HDF5 group is one complete synthetic experiment.

## Related Tutorial Notebooks

For notebook-first learning, use the focused notebooks under `tutorials/`:

- [`tutorial_hologram_pipeline_usage.ipynb`](../tutorials/tutorial_hologram_pipeline_usage.ipynb)
  walks through `HologramPipelineConfig`, `HologramPipelineRanges`, running the
  pipeline, reading HDF5 outputs, and visualizing masks, magnetic patterns,
  CR/CL holograms, CR-CL channels, and exit waves.
- [`tutorial_multislice_and_roi_modes.ipynb`](../tutorials/tutorial_multislice_and_roi_modes.ipynb)
  explains `propagate=True/False`, aperture ROI modes, Jones/tensor ROI,
  multislice ROI, padding, absorbers, and recommended operating modes.
- [`coherent_scattering_tutorials.md`](coherent_scattering_tutorials.md)
  maps all coherent-scattering notebooks by topic.

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

### Material Recipe

Recipe numbers are nanometres and layers are ordered top-to-bottom:

```python
recipe = "[Au(70)/Cr(30)]x10/SiN(200)/Co(90)/Pt(120)/Al(60)"
```

Slash-separated terms are propagated as separate layers. Adjacent terms without
a slash are combined into one effective-medium layer:

```python
Pt(4)/Co(6)   # two propagated layers
Pt(4)Co(6)   # one effective layer, 10 nm thick
```

For composite layers, the simulator averages the component dielectric tensor
channels by physical thickness. This preserves the existing isotropic, XMCD,
and XMLD tensor representation while reducing the number of free-space
propagation steps in multislice mode.

### ROI Mode

These flags reduce work by only generating or computing expensive quantities in
regions that matter:

```python
use_roi = True
dielectric_tensor_use_roi = True
dielectric_tensor_compact = True
```

`use_roi=True` enables OH-local magnetic patterns and local aperture/dielectric
optimizations. Turn it off to compute on the full slab.

`magnetic_pattern_use_roi=True` generates expensive texture patterns only in a
padded object-hole region and pastes that local pattern into a full field. This
is most useful for labyrinth or stripe-like patterns. Set it to `False` when
you want the full simulated field to contain the generated texture everywhere.

`dielectric_tensor_use_roi=True` creates aperture support regions and computes
magnetic/vacuum dielectric corrections only inside local aperture boxes. These
same support regions are also used by the Jones propagation fast path. If this
is disabled, Jones propagation has no aperture ROI boxes to use.

`dielectric_tensor_compact=True` avoids allocating the full dense
`(Nz, Ny, Nx, 2, 2)` tensor stack. The simulator stores constant per-layer
diagonal terms plus aperture ROI patches, then evaluates those patches during
Jones propagation. Set it to `False` if you need `final_dielectric_tensor` to be
a materialized dense array for debugging or downstream inspection.

#### What is approximated outside ROIs?

The ROI switches do not all mean the same thing, and none of them creates a hard
zero-valued mask for the exit wave.

`magnetic_pattern_use_roi=True` affects only magnetic-pattern generation for
supported expensive texture types, currently including labyrinth, wavy stripe,
and disordered skyrmion patterns. The pipeline creates a full magnetic-pattern
array initialized to the background value `+1`, generates the texture in a
padded object-hole bounding box, and pastes the local result into the full
array. Outside that box, the magnetic texture remains background. This assumes
texture outside the object-hole region is irrelevant because it is hidden by the
front aperture or by object-hole-focused diagnostics.

`dielectric_tensor_use_roi=True` affects the dielectric tensor and Jones
interaction. Aperture support boxes are built around the FTH holes. Inside those
boxes, the simulator evaluates the aperture, vacuum, magnetic, and off-diagonal
dielectric corrections. Outside those boxes, each material slice is approximated
as the spatially uniform diagonal background response for that layer. During the
Jones step, outside-ROI pixels are still multiplied by this constant per-layer
transmission; they are not zeroed.

`multislice_propagation_roi=True` affects only free-space propagation between
material slices, and only when `propagate=True`. The full field is first
advanced outside the ROI boxes by the zero-spatial-frequency plane-wave phase,
`exp(-i k0 dz)`. The local angular-spectrum FFT is then run in each padded
aperture ROI crop. The solver adds each crop's local correction,
`local_propagated - plane_wave_baseline`, into the full field. By default,
`multislice_propagation_roi_merge_overlaps=True`, so overlapping padded boxes
are merged before propagation and nearby apertures are propagated as one local
crop instead of letting separate crops compete in shared pixels. This can be
safer for close OH/RH layouts, but it can be slower because the merged crop is
larger. Set it to `False` only when you explicitly want the faster separate-crop
approximation.
Outside all boxes, no diffractive redistribution is computed. This is the
strongest ROI approximation because true free-space propagation is nonlocal:
diffracted light can move between ROI and non-ROI pixels.

If the exit-wave signal outside the plotted ROI rectangles is nonzero, that is
therefore expected. The rectangles show where expensive corrections or local
FFTs were evaluated, not where the field is allowed to exist. Use
`propagate=True` with `multislice_propagation_roi=False` for the conservative
full-field propagation reference, and increase
`multislice_propagation_roi_padding_px` when ROI multislice is useful but the
crop is too tight.

### Multislice Propagation

Free-space propagation between material layers is controlled near the top of
the script:

```python
propagate = True
multislice_propagation_roi = False
multislice_propagation_roi_padding_px = 64
multislice_propagation_roi_merge_overlaps = True
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

`multislice_propagation_roi=False` keeps the current full-field free-space
propagation. If set to `True`, the free-space FFT between slices is evaluated
only inside the aperture ROI boxes. The whole field first receives the
zero-spatial-frequency plane-wave phase, then each local ROI propagation adds
its correction relative to that baseline. This can be much faster for large
grids with small apertures, but it is approximate because true free-space
propagation is nonlocal and diffracted light can move between ROI and non-ROI
pixels.

`multislice_propagation_roi_padding_px` enlarges each aperture ROI box before
that approximate ROI-only free-space step. This is useful around small reference
holes, where a tight support box can truncate nearby diffracted structure. It is
separate from `propagation_padding_px`: ROI padding changes which pixels receive
full local propagation, while propagation padding only pads the FFT calculation
inside each crop and is cropped away afterward.

`multislice_propagation_roi_merge_overlaps=True` is the default. It merges
overlapping padded ROI boxes before local propagation. Use it when
reference-hole ROI boxes overlap the object-hole ROI and you want them treated
as one local diffraction crop. Set it to `False` only when you explicitly want
the faster separate-crop approximation.

Common operating modes:

- **Jones-only with aperture ROI**: `propagate=False`, `use_roi=True`,
  `dielectric_tensor_use_roi=True`, `dielectric_tensor_compact=True`.
- **Full-field multislice**: `propagate=True`,
  `multislice_propagation_roi=False`.
- **Multislice with Jones/tensor ROI only**: `propagate=True`, `use_roi=True`,
  `dielectric_tensor_use_roi=True`, `multislice_propagation_roi=False`.
- **ROI multislice, default merged crops**: `propagate=True`, `use_roi=True`,
  `dielectric_tensor_use_roi=True`, `multislice_propagation_roi=True`,
  `multislice_propagation_roi_merge_overlaps=True`.
- **ROI multislice, separate-crop opt-out**: same as ROI multislice, but set
  `multislice_propagation_roi_merge_overlaps=False` after checking that padded
  aperture boxes do not overlap or after validating the approximation against a
  full-field reference.
- **Full-field reference/debug mode**: `use_roi=False`.

For runnable comparisons and visual examples, see
[`tutorial_multislice_and_roi_modes.ipynb`](../tutorials/tutorial_multislice_and_roi_modes.ipynb).
That tutorial also includes a didactic vertical x-z wavefront diagnostic through
the aperture centers for each propagation mode. It overlays the aperture wall,
SiN membrane borders, magnetic-material borders, and the 1-D ROI propagation
intervals used by ROI multislice modes while showing amplitude and phase through
the stack. The diagnostic is reconstructed in the notebook for teaching; the
sweep HDF5 files still store the usual 2-D exit waves and holograms.

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

- `binary_labyrinth_pattern`: labyrinth domains generated from the binary
  Gray-Scott helper, rescaled to the requested stripe width.
- `skyrmion_pattern`: randomly placed circular skyrmions configured with
  `skyr_radius`, `screening_radius`, and a target count.
- `disordered_skyrmion_lattice_pattern`: random non-overlapping irregular
  skyrmions; `stripe_width` is the average skyrmion diameter.
- `image_pattern`: magnetic domains loaded from a binary experimental reconstruction.
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

`sigma` controls Gaussian domain-wall smoothing and is specified in metres in
pipeline configurations. `MagneticPatternConfig` converts it to pixels before
calling the selected generator. This conversion and smoothing apply to both
`skyrmion_pattern` and `disordered_skyrmion_lattice_pattern`, as well as the
labyrinth, stripe, and image-based generators.

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

The sweep also caps the random OH size relative to the magnetic texture length:

```python
max_oh_radius_in_texture_widths = 8.0
max_magnetic_pattern_roi_pixels = 768
```

This prevents unlucky combinations such as very small magnetic domains inside a
very large object hole, which can make the OH-local magnetic-pattern ROI large
and slow to generate. Increase these values when you intentionally want a large
object hole containing many magnetic domains.

### Experimental Image Patterns

The notebook and sweep script can use a binary experimental FTH reconstruction
as the base domain topology:

```python
pattern_type = "image_pattern"
experimental_pattern_path = DATA_ROOT / "Data" / "reconstruction_domains.png"
experimental_pattern_pixel_size = 5e-9  # metres per image pixel
experimental_pattern_threshold = 0.5
experimental_pattern_invert = False
experimental_pattern_pad_mode = "edge"
sigma = 30e-9
```

The loader converts RGB/RGBA images to grayscale, normalizes finite pixels to
`[0, 1]`, thresholds the result into `+1/-1` domains, rescales from
`experimental_pattern_pixel_size` to the simulation pixel size with
nearest-neighbour interpolation, then crops or pads to the simulated field of
view. The domain-wall width still comes from `sigma`, so the experimental image
sets the domain layout while the simulation controls the wall smoothing.

### Beamstop And Wire Mask

The beamstop can include a bent support wire:

```python
beamstop_config = {
    "radius": 0.2e-3,
    "sigma": 20e-6,
    "wire_width": 0.05e-3,
    "wire_bend": 0.1e-3,
    "antialias": 2,
}
```

The sweep script also randomizes the direct-beam detector position:

```python
detector_center=random_detector_center
```

`random_detector_center()` samples `(y, x)` in detector pixels within
`+/-5%` of the image size around the detector midpoint. The beamstop center
starts from that sampled detector center and then receives its own random
misalignment:

```python
beamstop_config=random_beamstop_config
```

The beamstop radius is still specified in metres at the beamstop plane. The
extra center offset is drawn in detector pixels after projecting the beamstop
radius to the detector plane, with the default range
`+/-0.5 * projected_beamstop_radius`.

`antialias` controls the beamstop and wire edge smoothing. Values above `1`
produce fractional mask values at edges. Use `1` for historical binary
rasterization.

By default, anti-aliased beamstops use `antialias_method="analytic"`, which
computes fractional edge coverage at detector resolution instead of allocating a
full supersampled detector. This is much faster for large images. For exact
historical subpixel averaging, add `antialias_method="supersample"` to
`beamstop_config`, but expect it to be substantially slower.

The important convention is that detector pixels store an average of the
continuous beamstop transmission over the pixel area. A beamstop edge, wire, or
subpixel object therefore produces fractional values instead of snapping to a
binary on/off value at the pixel center.

The normal detected hologram applies the beamstop before readout noise,
rounding, thresholding, and frame averaging. To additionally save a detected
hologram as if no beamstop were present, enable:

```python
save_detected_hologram_without_beamstop = True
```

This writes `CR/detected_no_beamstop` and `CL/detected_no_beamstop`. The optional
arrays are generated from the same photon and readout-noise realization as
`CR/detected` and `CL/detected`, but skip both the beamstop mask and detector
threshold cap.

### Skyrmion Parameters

The random skyrmion generator uses:

- `skyrmion_density`: approximate target skyrmion area fraction.
- `diameter_spread`: bounded half-range for diameter variation, in metres in the config.
- `ellipticity`: y/x axis-ratio range.
- `roughness`: boundary irregularity amplitude.
- `roughness_modes`: Fourier modes used for the rough boundary.

Each skyrmion gets its own random diameter, ellipticity, rotation angle, and
rough boundary. The disordered skyrmion lattice is rendered from a continuous
rough elliptical radial profile sampled onto the magnetic-pattern grid, rather
than from a binary pixel stamp that is blurred afterwards. This keeps very small
skyrmions subpixel-aware: they can have smooth partial-pixel edges instead of
collapsing into isolated binary pixels or cross-like shapes. The value assigned
to each magnetic-pattern pixel is the area average of the continuous skyrmion
profile over that pixel. A skyrmion occupying only part of one pixel therefore
has proportionally reduced contrast instead of full `-1/+1` contrast. `sigma`
is still specified in metres in the configuration, but after conversion to
pixels it is used as the skyrmion domain-wall transition width. If `sigma=0`,
the renderer integrates a hard continuous skyrmion boundary over the pixel
area.

The simpler `skyrmion_pattern` generator follows the same idea for circular
skyrmions: it samples a continuous circular core with a finite transition width
instead of drawing a hard binary circle and applying a global blur.

Candidate centers are generated from a jittered hexagonal proposal grid inside
the OH placement region, using the OH radius plus one average skyrmion diameter
as the placement radius. A center candidate is tried first so low-density
skyrmion samples still contain at least one skyrmion inside the OH field of
view. Candidates are accepted only if they do not overlap nearby accepted
skyrmions. A spatial hash keeps those overlap checks local, so dense cases do
not spend most of their time testing failed random attempts.

### Aperture Geometry And Roughness

The FTH mask contains one object hole (`OH`) and a random number of reference
holes (`RH`). Apertures are layer-aware:

- `aperture_top_radius_factors` controls the top/base radius ratio.
- The taper is derived from the material stack above the SiN membrane.
- The two layers closest to the SiN membrane remain cylindrical; if the stack
  above SiN is only two layers, the aperture stays cylindrical.
- Deeper layers below the taper keep the base aperture size.

Aperture holes use the same pixel-area averaging convention as beamstops and
skyrmions. The continuous aperture transmission is integrated over each
transverse pixel, so a reference hole smaller than one pixel becomes a partial
transmission change rather than a full binary pixel. `aperture_sigmas` are
continuous edge transition widths in metres after conversion to pixels, not
post-drawing Gaussian blurs.

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

### Detector, Measurement, And Artifact Ownership

These dictionaries configure different stages and should not repeat keys:

- `detector_params`: detector response and photon/count conversion:
  `readout_noise_average`, `readout_noise_sigma`, `detector_threshold`,
  `counts_per_photon`, and `quantum_efficiency`.
- `measurement_config`: acquisition timing and frame aggregation:
  `exposure_time`, `number_frames`, and `max_counts_per_image`.
- `artifacts_config`: photon-event shape and splatting controls such as
  `sigma_photon`, kernel size, class count, and irregularity.

Do not define one parameter in multiple dictionaries. Legacy aliases are
normalized when unambiguous; conflicting values raise `ValueError`.

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

`_pipeline_config/` contains only values that describe the whole file:

```text
_pipeline_config/
├── n_samples
└── oversampling
```

Each numbered sample group is self-contained. Its metadata records the effective
values actually used after applying sweep ranges and compatibility aliases:

```text
00000/
├── CR/                              # circular-right polarization
│   ├── exit_wave                    # complex sample-plane wavefield
│   ├── ideal                        # ideal detector hologram
│   ├── detected                     # noisy/artifact-affected hologram
│   └── detected_no_beamstop         # optional
├── CL/                              # same datasets for circular-left
├── beamstop_mask                    # detector-space beamstop transmission
├── supportmask                      # FTH aperture support
├── magnetic_pattern_oh              # magnetic pattern visible through the OH
└── metadata/
    ├── sample/
    │   ├── recipe, sample_name, real_space_pixel_size
    │   ├── use_roi                  # master sample/ROI switch
    │   ├── aperture/                # aperture method, thicknesses, geometry
    │   │   └── aperture_config/
    │   ├── magnetic_pattern/        # pattern method/config and ROI information
    │   └── dielectric_tensor/       # tensor ROI/compact flags
    ├── xray/                        # energy, flux, coherence, wavelength
    ├── detector/                    # detector geometry and output switches
    │   └── save_detected_no_beamstop
    ├── detector_params/             # readout noise, threshold, QE, counts/photon
    ├── measurement_config/          # exposure, frames, count normalization
    ├── artifacts_config/            # photon-event shape/splatting controls
    ├── beamstop/                    # effective beamstop geometry
    ├── illumination/                # beam profile, center, focus, FWHM
    ├── propagator_config/           # propagation method and effective config
    │   ├── propagator_method        # written first
    │   ├── propagate
    │   └── ...
```

Older files may also contain baseline simulation settings under
`_pipeline_config/`. New readers should use the numbered sample's `metadata/`
paths, because those are the effective values used for that sample.

### Dataset ownership

Each physical parameter has one canonical saved path. Important examples:

| Parameter | Canonical HDF5 path |
|---|---|
| X-ray energy | `metadata/xray/energy_eV` |
| Sample recipe | `metadata/sample/recipe` |
| Sample-plane pixel size | `metadata/sample/real_space_pixel_size` |
| Master ROI switch | `metadata/sample/use_roi` |
| Aperture geometry | `metadata/sample/aperture/aperture_config/...` |
| Magnetic-pattern method/config | `metadata/sample/magnetic_pattern/...` |
| Dielectric-tensor compact flag | `metadata/sample/dielectric_tensor/compact` |
| Detector distance | `metadata/detector/sample_to_detector_distance` |
| Readout-noise sigma | `metadata/detector_params/readout_noise_sigma` |
| Quantum efficiency | `metadata/detector_params/quantum_efficiency` |
| Counts per photon | `metadata/detector_params/counts_per_photon` |
| Exposure time | `metadata/measurement_config/exposure_time` |
| Photon-event sigma | `metadata/artifacts_config/sigma_photon` |
| Illumination FWHM | `metadata/illumination/fwhm_m` |
| Multislice enable flag | `metadata/propagator_config/propagate` |
| Multislice ROI flag | `metadata/propagator_config/multislice_propagation_roi` |

The hierarchy follows ownership:

- `sample/` owns the master ROI switch, aperture geometry, and dielectric
  tensor settings, and magnetic pattern because they describe how the sample is
  represented.
- `propagator_config/` owns both `propagator_method` and every propagation
  option. There is no separate `propagation/` or `propagator/` group.
  `propagator_method` is written first so a tree inspection identifies the
  selected algorithm before its options.
- `sample/use_roi` is the master switch. The scoped
  `sample/magnetic_pattern/use_roi` and
  `sample/dielectric_tensor/use_roi` datasets record whether each individual
  optimization was actually active. These similarly named datasets are
  intentional because they answer different questions.

Polarization datasets:

- `CR/ideal`, `CL/ideal`: ideal detector holograms before detector noise.
- `CR/detected`, `CL/detected`: detector holograms after noise/artifacts.
- `CR/detected_no_beamstop`, `CL/detected_no_beamstop`: optional detected
  holograms from the same noise realization as `detected`, but without beamstop
  masking or detector threshold capping.
- `CR/exit_wave`, `CL/exit_wave`: complex exit wavefields.

Other arrays:

- `beamstop_mask`: detector beamstop mask; anti-aliased edges can be fractional.
- `supportmask`: binary support mask in FTH reconstruction coordinates.
- `magnetic_pattern_oh`: magnetic pattern inside the OH save ROI; pixels outside the OH are zeroed.

`metadata/measurement_config`, `metadata/detector_params`, and
`metadata/artifacts_config` are deliberately separate sibling groups.
Measurement settings are not detector properties, and artifact-shape settings
do not own the counts-to-photon conversion.

`counts_per_photon` has one canonical location:
`detector_params/counts_per_photon`. It controls the conversion between detector
counts and photon events, including photon-artifact splatting. Do not also place
it in `artifacts_config`. Legacy artifact configurations containing the same
value are accepted as an input alias and normalized; conflicting values raise an
error instead of silently selecting one.

The same rule applies to detector noise and quantum efficiency:
`detector_params/readout_noise_sigma` and
`detector_params/quantum_efficiency` are canonical. The older `noise_rms`,
`detector_noise_rms`, and `detector_quantum_efficiency` inputs are compatibility
aliases only. If an alias conflicts with the canonical value, the pipeline
raises an error.

Magnetic-pattern physical lengths also belong directly in `pattern_config`.
`pattern_config_length` remains a legacy input alias, but conflicting keys raise
an error and it is not saved as a second metadata configuration.

Interactive simulations can use
`HologramPipeline.build_precomputed_metadata()` followed by
`HologramPipeline.write_precomputed_result()` to save an already-computed
`HologramConfig` with this same layout. The end of
`tutorials/Scattering_simulator_CK.ipynb` demonstrates this workflow.

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
    cr_detected_no_beamstop = (
        g["CR/detected_no_beamstop"][()]
        if "detected_no_beamstop" in g["CR"]
        else None
    )
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

    energy = m["xray/energy_eV"][()]
    pattern_type = m["sample/magnetic_pattern/pattern_type_method"][()].decode()
    detector_distance = m["detector/sample_to_detector_distance"][()]
    illumination_center = m["illumination/center_m"][()]
    aperture_roughness = m["sample/aperture/aperture_config/apertures_roughness"][()]
    propagate = m["propagator_config/propagate"][()]
    propagation_padding_px = m["propagator_config/propagation_padding_px"][()]
    propagation_padding_mode = m["propagator_config/propagation_padding_mode"][()].decode()
    propagation_absorber_width_px = m["propagator_config/propagation_absorber_width_px"][()]
    propagation_absorber_profile = m["propagator_config/propagation_absorber_profile"][()].decode()
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

print(meta["xray/energy_eV"])
print(meta["sample/magnetic_pattern/pattern_type_method"])
```

## Inspecting The Output

The final inspection section in `tutorials/simulate_hologram_sweep.py` opens
screen-friendly Matplotlib figures and uses `interpolation="none"` for every
`imshow` call so detector pixels and aperture cuts are shown without display
smoothing.

For a notebook that reads the HDF5 output and visualizes masks, aperture
geometry, magnetic pattern crops, CR/CL holograms, CR-CL/CR+CL channels,
reconstructions, and complex exit waves, see
[`tutorial_hologram_pipeline_usage.ipynb`](../tutorials/tutorial_hologram_pipeline_usage.ipynb).

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
  `metadata/sample/magnetic_pattern/saved_roi_*`.
- Callable range functions receive the already-sampled parameter dictionary, so
  they can build dependent ranges such as detector distance from energy and
  texture size.

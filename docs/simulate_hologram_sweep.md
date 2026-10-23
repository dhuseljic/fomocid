# Simulate Hologram Sweep

`tutorials/simulate_hologram_sweep.py` generates a sweep of simulated FTH
holograms and saves the result as one HDF5 file. Each simulated sample gets a
randomized x-ray energy, magnetic pattern, holography mask, illumination,
detector distance, beamstop, and detector acquisition settings.

The script is useful for producing training or benchmarking datasets where each
HDF5 group is one complete synthetic experiment.

## Related Tutorial Notebooks

For notebook-first learning, use the focused notebooks under `tutorials/`:

- [`09_hologram_pipeline.ipynb`](../tutorials/09_hologram_pipeline.ipynb)
  walks through `HologramPipelineConfig`, `HologramPipelineRanges`, running the
  pipeline, reading HDF5 outputs, and visualizing masks, magnetic patterns,
  CR/CL holograms, CR-CL channels, and exit waves.
- [`22_general_magnetic_domains.ipynb`](../tutorials/22_general_magnetic_domains.ipynb)
  explores the production domain controls, counts and order diagnostics.
- [`10_multislice_and_roi_modes.ipynb`](../tutorials/10_multislice_and_roi_modes.ipynb)
  explains `propagate=True/False`, aperture ROI modes, Jones/tensor ROI,
  multislice ROI, padding, absorbers, and recommended operating modes.
- [`11_tilted_magnetic_layer_multislice.ipynb`](../tutorials/11_tilted_magnetic_layer_multislice.ipynb)
  demonstrates tilted multilayers, fixed beam-direction contrast, and optional
  local-momentum vector contrast where XMCD follows the local light direction
  during multislice propagation.
- [`light_propagation_modes.md`](light_propagation_modes.md)
  derives the no-propagation approximation, multislice angular-spectrum
  propagation, final far-field FFT, and detector q-space projection.
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
3. Generates a resolved binary constrained-domain field, retains the precursor,
   and applies physical Gaussian wall smoothing on the native simulation grid.
4. Builds the holography support mask and detector beamstop.
5. Measures stripe/bubble counts and the
   `stripe`/`bubble`/`mixed`/`saturated` state inside the OH.
6. Computes the light-matter interaction for CR and CL polarization. Jones mode
   builds `final_dielectric_tensor`, a compact/dense `(Nz, Ny, Nx, 2, 2)`
   dielectric-tensor stack. Scalar mode builds `final_scalar_refractive_index`,
   a compact/dense `(Nz, Ny, Nx)` complex refractive-index stack.
7. Creates ideal holograms, applies detector effects/noise, and saves everything to HDF5.

By default, `verbose=True` prints experiment progress, a compact list of major-stage timings, the total
experiment time, skip/rejection notices, and the final summary. Set `detailed_logging=True` to
print per-stage timings. Magnetic generation
additionally prints the grid/pitch and live step progress, followed by a detailed
relaxation and wall-processing breakdown. All timings are also saved; see the
[timing reference](constrained_magnetic_domains.md#timing-and-live-generation-progress). `front aperture` reports
aperture-stack construction, assignment and detector-grid support generation;
`magnetic diagnostics` separately reports the OH-selected classification and
metadata work. Older logs charged both to `front aperture`. Diagnostics reuse
one component classification; periodic connectivity checks inspect only opposite
edges rather than sorting label pairs over the full image.

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
scalar_refractive_index_lazy = True
```

`use_roi=True` enables OH-local magnetic patterns and local aperture/dielectric
optimizations. Turn it off to compute on the full slab.

`magnetic_pattern_use_roi=True` generates expensive texture patterns only in a
padded object-hole region and pastes that local pattern into a full field. This
is most useful for labyrinth or stripe-like patterns. Set it to `False` when
you want the full simulated field to contain the generated texture everywhere.

`dielectric_tensor_use_roi=True` creates aperture support regions and computes
magnetic/vacuum optical corrections only inside local aperture boxes. In Jones
mode those corrections are dielectric-tensor patches. In Scalar mode the same
regions become scalar refractive-index patches. These support regions are also
used by the Jones and Scalar propagation fast paths. If this is disabled, the
propagators have no aperture ROI boxes to use.

`dielectric_tensor_compact=True` avoids allocating the full dense
`(Nz, Ny, Nx, 2, 2)` tensor stack. The simulator stores constant per-layer
diagonal terms plus aperture ROI patches, then evaluates those patches during
Jones propagation. Set it to `False` if you need `final_dielectric_tensor` to be
a materialized dense array for debugging or downstream inspection.

`scalar_refractive_index_lazy=True` is the Scalar-mode counterpart. Scalar mode
already avoids dielectric tensors; this flag also avoids precomputing every
scalar ROI patch. Instead, `final_scalar_refractive_index` stores the base
layer indices plus the mask, magnetization, and ROI boxes, then builds only the
current layer's scalar refractive-index patches during propagation. Set it to
`False` only when you want the compact scalar patches precomputed for
inspection or benchmarking.

#### What is approximated outside ROIs?

The ROI switches do not all mean the same thing, and none of them creates a hard
zero-valued mask for the exit wave.

`magnetic_pattern_use_roi=True` affects magnetic-pattern generation for the
constrained-domain texture used by this sweep. The pipeline creates a full magnetic-pattern
array initialized to the background value `+1`, generates the texture in a
padded object-hole bounding box, and pastes the local result into the full
array. For the constrained generator, this box follows the actual OH support,
with half a characteristic length plus four wall sigmas of context on each
side by default (`magnetic_pattern_context_periods=0.5`), at least two repeat
lengths per axis, and FFT-friendly dimensions. This avoids solving the texture
on an unnecessarily large tapered-aperture footprint. The fine simulation pitch
is preserved. The periodic box size affects the realized texture and is recorded
with its ROI bounds and generation shape. Outside that box, the magnetic texture remains background. This assumes
texture outside the object-hole region is irrelevant because it is hidden by the
front aperture or by object-hole-focused diagnostics.

`dielectric_tensor_use_roi=True` affects the dielectric tensor and Jones
interaction. Aperture support boxes are built around the FTH holes. Inside those
boxes, the simulator evaluates the aperture, vacuum, magnetic, and off-diagonal
dielectric corrections. Outside those boxes, each material slice is approximated
as the spatially uniform diagonal background response for that layer. During the
Jones step, outside-ROI pixels are still multiplied by this constant per-layer
transmission; they are not zeroed.

`dielectric_tensor_local_k_projection=True` disables compact/ROI tensor storage
for Jones mode and stores a direction-independent vector dielectric response
instead. At every material slice the propagator estimates the local wavevector
direction from the current Jones field, projects the magnetic response onto
that direction, and then applies the resulting local Jones matrix. This is the
most physical mode for thick tilted samples or aperture stacks where the light
has diffracted before it reaches a magnetic layer. It is also more expensive,
because the projected tensor image is rebuilt for each slice from the evolving
field.

`multislice_propagation_roi=True` affects only free-space propagation between
material slices, and only when `propagate=True`. The full field is first
advanced outside the ROI boxes by the zero-spatial-frequency plane-wave phase,
`exp(-i k0 dz)`. The local angular-spectrum FFT is then run in each padded
aperture ROI crop. The solver adds each crop's local correction,
`local_propagated - plane_wave_baseline`, into the full field. Jones and Scalar
use the same propagation convention here: local FFT crops use
`exp(-i kz dz)`, while the outside baseline is the `kz = k0` limit of the same
operator. By default,
`multislice_propagation_roi_merge_overlaps=True` encloses all padded aperture
boxes in one common local propagation crop, so apertures can diffract into one
another inside that crop without paying for a full-field FFT. If you set the
flag to `False`, physically separate and disjoint padded boxes may remain
separate for speed. Aperture supports that overlap in the actual material mask
are always merged before padding; this is mandatory because intersecting
funnels form one physical hole system. Padded boxes that overlap are also
merged before propagation, even when the flag is `False`; otherwise the same
pixels can receive multiple local FFT corrections and become nonphysically
bright. The local correction is smoothly tapered across the ROI padding before
it is added back to the full field, which reduces vertical/horizontal bands
from hard rectangular crop edges. Set the flag to `False` only when you
explicitly want the faster separate-crop approximation between disjoint padded
boxes.
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
propagator_method = "Jones"
scalar_refractive_index_lazy = True
jones_apply_zero_order_phase = True
multislice_propagation_roi = False
multislice_propagation_roi_padding_px = 64
multislice_propagation_roi_merge_overlaps = True
propagation_padding_px = 128
propagation_padding_mode = "edge"
propagation_absorber_width_px = 64
propagation_absorber_strength = 6.0
propagation_absorber_profile = "cosine"
farfield_oversampling = 1
```

`propagator_method="Jones"` is the default full two-component Jones-matrix
light-matter interaction. `propagator_method="Scalar"` is the faster scalar
alternative: it skips dielectric-tensor construction and builds a compact
scalar refractive-index stack, `final_scalar_refractive_index`, directly from
the database channels
`[n_total, n_circ, n_lin]`, the aperture mask, and the magnetization. In the
eigenmode case, where the chosen polarization is a local eigenvector of the
interaction matrix, this scalar refractive-index interaction matches Jones
propagation while avoiding two-component matrix propagation. Use Jones when the
sample can rotate or mix polarization components in a way you need to keep.
With `scalar_refractive_index_lazy=True`, the scalar ROI patches are computed
layer by layer while propagating instead of being allocated for the whole stack
before propagation. Scalar and Jones use the same free-space phase convention
for full-field and ROI multislice propagation, so Scalar ROI crops should not
introduce a phase jump relative to the plane-wave baseline.

When `propagate=True`, the field is propagated between consecutive material
layers with an angular-spectrum free-space propagator using the actual sample
pixel size and each layer thickness. This is more physical for thick or
strongly structured aperture stacks, but it is slower because it adds FFTs
between layers. When `propagate=False`, the simulator applies only the local
material transmission for each layer, which is the faster historical mode. By
default, `jones_apply_zero_order_phase=True` still multiplies the field by the
rank-zero free-space factor `exp(-1j * k0 * dz)` between layers. For scalar
propagation the same setting is forwarded as `scalar_apply_zero_order_phase`.
This preserves the longitudinal phase advance without evaluating transverse
FFT diffraction. Set it to `False` only when reproducing older calculations
that omitted this inter-layer phase.

For the propagation equations, including the no-propagation approximation,
full-field multislice FFTs, ROI multislice corrections, final Fraunhofer FFT,
and detector q-space projection, see
[`light_propagation_modes.md`](light_propagation_modes.md).

### Vector Contrast And Local Momentum

For the derivation of charge, XMCD, XMLD, scalar refractive-index propagation,
Jones dielectric tensors, fixed beam-direction projection, and local-k vector
contrast, see
[`optical_contrast_formalisms.md`](optical_contrast_formalisms.md). The short
version for this sweep script is:

- `propagator_method="Scalar"` uses one effective complex refractive index for
  the selected polarization eigenmode.
- `propagator_method="Jones"` propagates the two-component transverse field
  through a dielectric tensor.
- tilted Jones illumination can project XMCD onto the nominal beam direction,
  so contrast follows `m . k` instead of lab-frame `mz`;
- `dielectric_tensor_local_k_projection=True` recomputes `k` from local Jones
  phase gradients before every material slice, so multislice diffraction can
  change the vector contrast through the stack.

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
inside each crop and is cropped away afterward. The ROI correction is multiplied
by a smooth raised-cosine window across this padding before it is pasted back
into the full field, so the correction returns gradually to the plane-wave
baseline instead of stopping at a hard rectangular edge.

`multislice_propagation_roi_merge_overlaps=True` is the default. The code always
merges aperture supports whose funnels overlap in the material mask, because
those pixels belong to the same physical hole system. Padded ROI boxes that
overlap are also forced to merge before local propagation. This boolean is
therefore only a speed/accuracy preference for disjoint padded boxes. Use the
default for close OH/RH layouts. Set it to `False` only when you explicitly want
the faster separate-crop approximation between disjoint padded boxes.

Common operating modes:

- **Jones-only with aperture ROI**: `propagate=False`, `use_roi=True`,
  `dielectric_tensor_use_roi=True`, `dielectric_tensor_compact=True`,
  `jones_apply_zero_order_phase=True`.
- **Scalar eigenmode interaction**: set `propagator_method="Scalar"` when the
  selected polarization is expected to remain a local eigenmode. This skips
  dielectric-tensor construction, uses the database refractive-index channels
  directly, and can be substantially faster than Jones while matching it for
  diagonal or circular-eigenmode stacks. Keep
  `scalar_refractive_index_lazy=True` to compute scalar patches layer by layer
  and minimize memory allocation. The scalar free-space propagator uses the
  same `exp(-i kz dz)` angular-spectrum convention as Jones.
- **Full-field multislice**: `propagate=True`,
  `multislice_propagation_roi=False`.
- **Multislice with Jones/tensor ROI only**: `propagate=True`, `use_roi=True`,
  `dielectric_tensor_use_roi=True`, `multislice_propagation_roi=False`.
- **ROI multislice, default common crop**: `propagate=True`, `use_roi=True`,
  `dielectric_tensor_use_roi=True`, `multislice_propagation_roi=True`,
  `multislice_propagation_roi_merge_overlaps=True`. This uses one common crop
  enclosing all padded aperture boxes.
- **ROI multislice, separate-crop opt-out**: same as ROI multislice, but set
  `multislice_propagation_roi_merge_overlaps=False` after checking that the
  padded boxes are disjoint and after validating the approximation against a
  full-field reference.
- **Full-field reference/debug mode**: `use_roi=False`.

For runnable comparisons and visual examples, see
[`10_multislice_and_roi_modes.ipynb`](../tutorials/10_multislice_and_roi_modes.ipynb).
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
- `farfield_oversampling`: factor used to extend the complex exit wave before
  the far-field FFT. Values above `1` do **not** zero-pad the field. Instead,
  the larger field is filled with the configured illumination multiplied by the
  uniform background dielectric stack and the ROI plane-wave free-space factors,
  then the simulated central exit wave is pasted into the middle. This gives a
  finer reciprocal-space hologram grid without inventing a black boundary around
  the sample.

The previous per-polarization implementation is kept in
`propagate_free_space_jones_260526` for future comparisons.

### Magnetic Patterns

The sweep uses `pattern_type="constrained_domain_pattern"`, the same constrained
scalar model explored in Notebook 22. It samples physical spacing, net
magnetization, alignment and angle, disorder, initial preparation, solver steps,
calibration controls, wall sigma and seed through
`sample_constrained_domain_parameters(rng)`.

There are no requested saturated/stripe/bubble handles, phase-space lookup cache
or balanced morphology quotas. Saturation arises at `target_magnetization=±1`;
all other state labels come from measured connected-domain topology. The script
now draws 48% free parents, 48% Gaussian-target parents (mean 0, FWHM 0.5,
bounds ±0.9), and 4% saturated parents. Adjust the three probability settings
to change this mixture.
Measured full-box and OH means are saved in every mode. The sweep
runs one relaxed realization (`max_length_iterations=1`) and measures achieved
spacing and order afterward. It does not repeatedly regenerate patterns to hit
target measurements. `characteristic_length` is the preferred model scale;
`length_target_met` is diagnostic only. The inner relaxation steps still form
the domain texture, and wall blur is still applied before optical propagation.

```python
pattern_type = "constrained_domain_pattern"
pattern_config = dict(
    characteristic_length=600e-9, target_magnetization=0.,
    alignment=0., disorder=0., initial_state="random", initial_order=0.,
    n_steps=300, max_length_iterations=1, length_tolerance=.15,
    sigma=30e-9, seed=7,
)
```

Read the [constrained magnetic-domain reference](constrained_magnetic_domains.md#production-hologram-generation)
for the complete parameter table, shipped sampling distributions, wall treatment,
resolution limits, state rules, order statistics and HDF5 paths.

Binary domains are generated on the native fine simulation grid. Gaussian
wall blur is then applied using physical `sigma` (metres). Classification uses
the retained binary precursor rather than the blurred sign boundary. The optics
uses the smoothed magnetization and the existing transverse wall component.

### Pattern Size Parameter

`characteristic_length` is the target **full repeat/spacing**, not a stripe width,
bubble diameter or worm length. `target_magnetization` fixes full-grid sign area when numeric; `None` lets it evolve freely.
Both achieved spacing and OH-observed occupancy are measured separately.
All lengths in the pipeline are metres; direct generator calls use pixels.

To prevent blocky source patterns, the hologram backend raises actual integer
oversampling to resolve the spacing by at least 16 pixels and positive Gaussian
wall sigma by at least 2 pixels, plus four pixels across the nonzero conservative
minority stripe-width estimate. It records per-sample
`metadata/sample/effective_oversampling`. The script samples detector distance uniformly from 5 to 40 cm
(`Uniform(.05, .40)` in metres), once per independent experiment. If the necessary full-grid axis exceeds
`magnetic_pattern_max_grid_axis` (default 4096), the draw is rejected before
allocating the large grid. The runner resamples the complete experiment up to
`max_sampling_attempts=20` times, then skips that experiment and continues. It does not silently reduce resolution. Explicit-grid experiments
must select a sufficiently fine pitch themselves. Refine further to establish
numerical convergence for a particular sample. Far-field padding does not
improve the spatial resolution of magnetic walls.

Legacy pattern types remain available when explicitly selected, including
`binary_labyrinth_pattern`, `wavy_stripe_pattern`, `skyrmion_pattern`,
`disordered_skyrmion_lattice_pattern`, `saturated_pattern` and `image_pattern`.
Their generator-specific controls do not apply to the constrained model.

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

Ideal holograms are projected from the simulated reciprocal-space grid onto the
detector grid with positivity-preserving linear interpolation by default:

```python
use_detector_pixel_footprint = False
detector_pixel_footprint_samples = 3
ignore_flat_detector_curvature = False
```

Set `use_detector_pixel_footprint=True` in the sweep file to average each ideal
detector pixel over a regular
`detector_pixel_footprint_samples x detector_pixel_footprint_samples`
sub-sampling grid. This is closer to a finite detector-pixel footprint and
cannot create negative intensity values from a non-negative hologram, but it is
slower by roughly the number of sub-samples per pixel. The package default
`False` mode uses one centre sample per detector pixel and is faster.

Set `ignore_flat_detector_curvature=True` to project detector pixels with the
linear reciprocal-space approximation `qx = k * x / z` and `qy = k * y / z`
instead of the default flat-detector angular mapping
`q = k * sin(arctan(r / z))`. The default `False` preserves the previous
hologram outputs and includes the nonlinear q spacing caused by a flat detector.

In both modes, the projection always applies the flat-detector solid-angle
collection factor. A pixel at detector-plane coordinate `(x, y)` and
sample-detector distance `z` is multiplied by
`(z / sqrt(x**2 + y**2 + z**2))**3`, so off-axis pixels collect fewer photons
than an on-axis pixel with the same simulated intensity per solid angle.

The normal detected hologram applies the beamstop to expected photons before
photon sampling, charge spreading and response, and therefore before readout noise,
rounding, thresholding, and frame averaging. To additionally save a detected
hologram as if no beamstop were present, enable:

```python
save_detected_hologram_without_beamstop = False  # default; set True to enable
```

This writes `CR/detected_no_beamstop` and `CL/detected_no_beamstop`. The optional
arrays are generated from the same photon and readout-noise realization as
`CR/detected` and `CL/detected`, but skip both the beamstop mask and detector
threshold cap.

### Library-only Skyrmion Parameters

The current sweep does not select the handcrafted skyrmion generators. The
following parameters remain available to callers of the reusable pattern
library and to older custom configurations:

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
holes (`RH`). It can also contain rectangular slits (`SLIT`). Apertures are
layer-aware:

- `aperture_top_radius_factors` controls the top/base radius ratio.
- For `SLIT` apertures, `aperture_radii` is the slit width and
  `aperture_lengths` is the slit length.
- The taper is derived from the material stack above the SiN membrane.
- The two layers closest to the SiN membrane remain cylindrical; if the stack
  above SiN is only two layers, the aperture stays cylindrical.
- Deeper layers below the taper keep the base aperture size.

Aperture holes use the same fractional-pixel convention as beamstops and
skyrmions. The continuous aperture transmission is estimated from a native-grid
signed-distance model, so reference-hole edges and holes smaller than one pixel
become partial transmission changes rather than full binary pixels. This avoids
the older expensive subpixel rasterization loop. `aperture_sigmas` are
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
# Optional beam tilt (alpha_y, alpha_x). Keep (0.0, 0.0) for normal incidence.
# illumination_alpha_beam=(Uniform(-np.deg2rad(2), np.deg2rad(2)), Uniform(-np.deg2rad(2), np.deg2rad(2)))
illumination_center=(
    Uniform(-2e-6, 2e-6),
    Uniform(-2e-6, 2e-6),
)
```

The center tuple is `(y, x)` in metres. `illumination_alpha_beam` is
`(alpha_y, alpha_x)` in radians. It evaluates the Gaussian beam on a sample
plane tilted in the two sample-plane directions with respect to the beam normal;
`(0.0, 0.0)` uses the exact historical normal-incidence illumination. For
backward compatibility, a scalar value is still accepted and interpreted as
`(0.0, scalar)`.

### Detector, Measurement, And Artifact Ownership

These dictionaries configure different stages and should not repeat keys:

- `detector_params`: detector response and photon/count conversion:
  `readout_noise_average`, `readout_noise_sigma`, `detector_threshold`,
  `counts_per_photon`, and `quantum_efficiency`.
- `measurement_config`: acquisition timing and frame aggregation:
  `exposure_time`, `number_frames`, and `max_counts_per_image`.
- `artifacts_config`: photon-event splatting plus camera defects and cosmic-ray
  controls. This includes the persistent `camera_seed`, hot/cold count means,
  pixel-value variation, cosmic-ray rate, intensity, size, and aspect ratio.

Do not define one parameter in multiple dictionaries. Legacy aliases are
normalized when unambiguous; conflicting values raise `ValueError`.

### Linear and nonlinear detector response

The sweep explicitly exposes `detector_params["nonlinear_response"] = None`
and `measurement_config["acquisition_mode"] = "accumulated"`. Keep these defaults
for linear, fast acquisition. A response callable maps per-frame linear ADU to
nonlinear ADU; `counts_per_photon` can also be a physical-energy gain callback.
Both are passed through to the detector, not evaluated as sweep factories.
Selecting `per_frame` applies response and clipping separately for each frame.
A linear curve still retains readout noise, defects and ADC clipping.

The HDF5 metadata records `detector_response/model`, callback identifiers and
actual per-helicity energy/gain values. See the
[detector-response guide](detector_response.md#paired-cr-cl-hologram-sweep) for
configuration examples, units, acquisition-mode details and exact HDF5 paths.

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

### Controlling Hot, Cold, And Cosmic-Ray Artifacts

Hot, cold, and flickering pixels belong to a simulated camera. Reusing
`camera_seed` reproduces both their `(y, x)` coordinates and their per-pixel
baseline values. The controls have two distinct kinds of variation:

- `hot_pixel_value` and `cold_pixel_value` are the central detector-count
  values.
- `hot_pixel_value_spread` and `cold_pixel_value_spread` are fractional
  standard deviations used once, when the camera seed assigns a different
  baseline to every defective pixel.
- `hot_pixel_temporal_sigma` and `cold_pixel_temporal_sigma` are fractional
  standard deviations around each assigned baseline in every exposure.
- `flicker_fraction` selects which hot-pixel coordinates are unstable, and
  `flicker_probability` controls their per-frame activation probability.

Cosmic rays are independent exposure events. Their expected count is
`cosmic_rays_per_second * exposure_time`; a missing exposure uses 1 second.
Each event is rendered as a rotated, smoothly blurred Gaussian ellipse:

- `cosmic_ray_value` is the central peak count and
  `cosmic_ray_value_spread` varies the peak independently per event.
- `cosmic_ray_length_range` is the sampled major-axis FWHM range in pixels.
  Short values such as `(2.0, 5.0)` avoid long line-like tracks.
- `cosmic_ray_aspect_ratio_range` is sampled independently for every event.
  `(2.0, 3.0)` gives varied compact ellipses within the requested ratio range.

The production sweep samples these settings per simulated camera. CR and CL
share that camera map but use different exposure seeds, so their cosmic rays
and temporal pixel oscillations differ.

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
├── beamstop_mask                    # detector-space blocked fraction (1 = blocked)
├── supportmask                      # FTH aperture support
├── magnetic_pattern_oh              # magnetic pattern visible through the OH
└── metadata/
    ├── sample/
    │   ├── recipe, sample_name, real_space_pixel_size
    │   ├── use_roi                  # master sample/ROI switch
    │   ├── aperture/                # aperture method, thicknesses, geometry
    │   │   └── aperture_config/
    │   ├── magnetic_pattern/        # phase-space config, OH-local state/counts, ROI
    │   └── dielectric_tensor/       # tensor ROI/compact flags
    ├── xray/                        # energy, flux, coherence, wavelength
    ├── detector/                    # detector geometry and output switches
    │   ├── save_detected_no_beamstop
    │   ├── use_detector_pixel_footprint
    │   ├── detector_pixel_footprint_samples
    │   └── ignore_flat_detector_curvature
    ├── detector_params/             # readout noise, threshold, QE, counts/photon
    ├── measurement_config/          # exposure, frames, count normalization
    ├── artifacts_config/            # requested camera/cosmic-ray controls
    ├── artifacts_realized/
    │   ├── camera/                  # defect coordinates and baseline values
    │   ├── CR/                      # realized CR cosmic-ray events
    │   └── CL/                      # realized CL cosmic-ray events
    ├── beamstop/                    # effective beamstop geometry
    ├── illumination/                # beam profile, center, focus, FWHM, tilt
    ├── propagator_config/           # propagation method and effective config
    │   ├── propagator_method        # written first
    │   ├── propagate
    │   ├── jones_apply_zero_order_phase
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
| Measured magnetic state | `metadata/sample/magnetic_pattern/state` |
| Bubble count in OH | `metadata/sample/magnetic_pattern/bubble_count` |
| Stripe-like count in OH | `metadata/sample/magnetic_pattern/stripe_count` |
| Sub-resolution component count | `metadata/sample/magnetic_pattern/noise_component_count` |
| Dielectric-tensor compact flag | `metadata/sample/dielectric_tensor/compact` |
| Fixed beam projection flag | `metadata/sample/dielectric_tensor/beam_direction_projected` |
| Fixed beam direction | `metadata/sample/dielectric_tensor/beam_direction_xyz` |
| Local-k projection flag | `metadata/sample/dielectric_tensor/local_k_projected` |
| Detector distance | `metadata/detector/sample_to_detector_distance` |
| Readout-noise sigma | `metadata/detector_params/readout_noise_sigma` |
| Quantum efficiency | `metadata/detector_params/quantum_efficiency` |
| Counts per photon | `metadata/detector_params/counts_per_photon` |
| Exposure time | `metadata/measurement_config/exposure_time` |
| Photon-event sigma | `metadata/artifacts_config/sigma_photon` |
| Camera seed | `metadata/artifacts_config/camera_seed` |
| Hot/cold coordinates | `metadata/artifacts_realized/camera/*_pixels_yx` |
| Hot/cold baselines | `metadata/artifacts_realized/camera/*_baseline_values` |
| Cosmic-ray event rows | `metadata/artifacts_realized/{CR,CL}/cosmic_ray_events_yx_length_aspect_angle_value` |
| Illumination FWHM | `metadata/illumination/fwhm_m` |
| Illumination beam tilt | `metadata/illumination/alpha_beam_rad` (`alpha_y`, `alpha_x`) |
| Multislice enable flag | `metadata/propagator_config/propagate` |
| Jones-only rank-zero phase flag | `metadata/propagator_config/jones_apply_zero_order_phase` |
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

### OH-local magnetic morphology metadata

The classifier labels connected domains of **both polarities on the full binary
precursor**, then counts a component once if it intersects the OH. It uses
pixels outside the aperture to establish shape; a bubble clipped by the OH
remains a bubble. The constrained generator is periodic, so opposite edges of
the full generated grid (or generation ROI) are joined before measuring shape.
A component is counted once only if it has at least one pixel in the OH,
even when it appears at multiple edges. Entirely hidden domains are excluded
from counts and order. No clipped-bubble fragments are discarded in this mode.
`classification_periodic=True` records this convention. Explicit legacy
nonperiodic patterns retain open-boundary classification.

Defaults are minimum full component area 9 pixels, maximum eccentricity 0.95
(approximately 3.2:1 axis ratio), minimum digital circularity 0.4 and eight-neighbor
connectivity. Other resolved components, including connected majority
backgrounds, are stripe-like. Noise is counted separately. Positive/negative and
majority/minority counts are saved alongside their sums.

| Total bubble count B / stripe-like count S | `state` |
|---|---|
| B>0, S=1 | `bubble` |
| B=0, S>1 | `stripe` |
| B>0, S>1 | `mixed` |
| B=0, S=1 | `saturated` |
| B>0, S=0 (background outside OH) | `bubble` |
| B=0, S=0 | `unresolved` |

A homogeneous pattern counts as one stripe-like component. A single connected
labyrinth counts as one domain rather than as multiple branches. Inspect
`noise_count`/`is_uniform` before equating the state label with exact uniformity.
The reusable `morphology` descriptor describes the component mixture and can
therefore differ from `state` (for example, bubbles plus one background).

Under `metadata/sample/magnetic_pattern/`, the file retains `generation_json`
(including calibration and relaxation traces),
`classification_json` (including both-polarity component records),
`measurements_json` and named scalar `measurements/` datasets. Effective physical
inputs remain under `pattern_config/`. `magnetic_pattern_oh` stores the blurred
optical magnetization, possibly on a different saved ROI.

Order diagnostics include component-axis `stripe_nematic_order` and local/global
sixfold bubble-center bond order (`local_sixfold_order`, `sixfold_order`). They
are orientational statistics rather than proofs of crystalline periodicity;
undefined values are JSON null or numeric NaN. Full definitions, limitations,
sampling conventions and HDF5 paths are in the
[domain reference](constrained_magnetic_domains.md#diagnostics-and-hdf5-provenance).

```python
import json
import h5py
with h5py.File("simulation_sweep.h5", "r") as h5:
    analysis = json.loads(h5["00000/metadata/sample/magnetic_pattern/classification_json"].asstr()[()])
    print(analysis["state"], analysis["bubble_count"], analysis["stripe_count"])
```

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
`tutorials/legacy/ck_interactive_monolith.ipynb` demonstrates this workflow.

## Reading The HDF5 File

For collaborator analysis, use the [magnetic-pattern analysis guide](magnetic_pattern_analysis.md).
It provides a loader, physical-coordinate plots, per-specimen CSV export and
precise definitions of requested versus achieved values, periodic OH-selected
counts, state labels, component diagnostics, wall smoothing and order statistics.

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
    magnetic_state = m["sample/magnetic_pattern/state"][()].decode()
    bubble_count = m["sample/magnetic_pattern/bubble_count"][()]
    stripe_count = m["sample/magnetic_pattern/stripe_count"][()]
    detector_distance = m["detector/sample_to_detector_distance"][()]
    illumination_center = m["illumination/center_m"][()]
    aperture_roughness = m["sample/aperture/aperture_config/apertures_roughness"][()]
    propagate = m["propagator_config/propagate"][()]
    propagation_padding_px = m["propagator_config/propagation_padding_px"][()]
    propagation_padding_mode = m["propagator_config/propagation_padding_mode"][()].decode()
    propagation_absorber_width_px = m["propagator_config/propagation_absorber_width_px"][()]
    propagation_absorber_profile = m["propagator_config/propagation_absorber_profile"][()].decode()
    illumination_alpha_beam = m["illumination/alpha_beam_rad"][()]
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
[`09_hologram_pipeline.ipynb`](../tutorials/09_hologram_pipeline.ipynb).

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

The full transformed array is the Patterson map. An FTH image is obtained by
cropping an object/reference-hole cross-correlation lobe at the displacement
set by the reference-hole position; the bright central autocorrelation is not
the magnetic reconstruction. Keep that distinction when evaluating downstream
masking or reconstruction algorithms.

`beamstop_mask` is a detector-plane obstruction map: values near `1` are
blocked and its transmission is `1 - beamstop_mask`. Because `CR/detected` and
`CL/detected` already include this shadow, applying a predicted beamstop mask to
those arrays again is mostly redundant. To compare a predicted mask with the
ideal mask on a practical FTH task, apply both masks separately to the same
`CR/detected_no_beamstop - CL/detected_no_beamstop` input, reconstruct both,
crop the same cross-correlation lobe, and compare with shared display limits or
a quantitative error metric. Do not independently percentile-normalize the two
images when judging their difference.

## Common Edits

Change output size:

```python
nr_simulations = 1000
```

Turn off reproducibility:

```python
pipeline_random_seed = None
```

Change the sampled domain controls in
`sample_constrained_domain_parameters(rng)`, or replace `random_pattern_config`
with a callable returning fixed or sampled constrained-model parameters. For
example, fix `target_magnetization=-0.6` and vary alignment and disorder; the
final state is still measured after generation. Keep
`pattern_type="constrained_domain_pattern"`.

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

See [shared magnetic controls](experiment_setup.md#shared-magnetic-pattern-controls) for names, units and compatibility aliases.

For branching multiple energies/masks, camera geometries and detector acquisitions
from shared upstream results, use the separate
[hierarchical sweep](hierarchical_hologram_sweep.md).

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
sample distribution: the proposed detector distance remains uniform from 5 to
40 cm, but the saved distances are conditioned on grid feasibility. In a tree,
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


For interactive configuration and plotting of one experiment at a time, use the
[desktop simulator](single_experiment_gui.md).

### Experiment timeout

The example sets `experiment_timeout_seconds=100.0` next to the simulation count.
Each experiment runs in an isolated process with a temporary output file. If it
exceeds 100 seconds, the child is stopped, all its temporary data is deleted,
and the sweep moves to the next experiment. Completed experiments remain intact.
No partial group or per-experiment timeout record is saved. Only successful
results are copied into the main HDF5 file. The existing summary counts reflect
how many experiments completed or were skipped. Numeric experiment IDs can have
gaps; the example plots the first completed group and exits cleanly if none
completed.

The deadline covers computation and temporary output, including all grid-retry
attempts for that experiment. Copying a completed temporary result into the
main dataset is outside that deadline. The constructor accepts another positive
limit, or `None` to disable isolation/timeouts. This timeout implementation
requires Unix `fork` support and works with the example's callable samplers;
parameter draws happen in the parent process so random generators advance even
when an experiment is killed. Failed runs are not retried after a timeout.

Sweep output defaults to major-step timings and total elapsed time
(`verbose=True, detailed_logging=False`).
Set `detailed_logging=True` to restore generator, detector and stage breakdowns.
All timing diagnostics are still saved in HDF5. The desktop GUI enables detailed
logging in its dedicated progress pane.

### Shared nonmagnetic Scalar propagation

For the default lazy Scalar backend, a paired CR/CL calculation reuses the
common isotropic prefix. AuCr and SiN are propagated once; the calculation
splits before the first layer with a nonzero circular or linear anisotropic
optical channel (CoXY in the example recipe). Each helicity still computes its
own magnetic suffix, final far-field transform, ideal projection and acquisition.
Free-space diffraction or longitudinal phase between the common layers is part
of the reused field. Incident fields, material prefix and propagation settings
must match; otherwise reuse is skipped. The cache is restricted to the current
unchanged sample pair and is never written to HDF5. Per-helicity counters are
saved at `metadata/propagator/{CR,CL}/reused_prefix_layers`.

Dense/non-lazy Scalar stacks, per-layer callbacks, Jones and Stokes retain their
independent propagation paths. In Jones propagation the CR and CL vectors are
different even through isotropic layers; sharing their scalar envelope would
require a separate vector-aware implementation. Merely looking for off-diagonal
matrix entries is insufficient when other anisotropic optical terms are present.

### Mask reuse and output compression

The physical 3-D aperture mask is constructed once per sample, before the
helicity loop. Lazy Scalar setup also reuses the aperture-support geometry for
CR and CL; only their polarization-dependent optical terms differ. A redundant
pre-loop CR scalar-index preparation has been removed. Geometry caches are
scoped to the unchanged sample pair and are not serialized.

`HologramPipelineConfig.hdf5_compression` defaults to **`"lzf"`**, a fast lossless
HDF5 filter. Set it to `"gzip"` for typically smaller files with slower writes,
or `None` for uncompressed arrays. Existing readers and dataset paths remain
compatible. The selected filter is recorded in the pipeline configuration.
The single-experiment GUI exposes the same control under **Output**. The
independent and hierarchical hologram pipelines share this writer; the separate
`simulate_experiment` backend retains its own output-compression setting.

A representative pair of 1536×1536 complex64 exit waves took median 1.08 s to
write with gzip (28.8 MB), 0.20 s with LZF (36.8 MB), and 0.023 s uncompressed
(37.8 MB), across three local trials. Complete experiment writes also include
other arrays and metadata, and performance depends on storage and data values.

See [Choosing the HDF5 save mode](hdf5_save_modes.md) for script, command-line
and GUI controls, storage precision, lossless roundtrip checks and readback.

The optional no-beamstop detected output is disabled by default in both examples
and the GUI. Set `save_detected_hologram_without_beamstop=True` to generate and
save it; while disabled, its complementary photon events and image are not computed.


The compact Scalar propagation summary also prints one line with **shared prefix**,
**CR suffix**, **CL suffix**, and **final FFTs** times. For the example recipe,
the suffix is the final CoXY layer; for other recipes it includes everything
from the first polarization-dependent layer onward. The prefix is computed
once, so the cached CL pass records zero prefix computation time. Per-helicity
values are saved at `metadata/propagator/{CR,CL}/timing_seconds/` under
`shared_prefix`, `polarization_suffix`, and `farfield`. The propagation-stage
total additionally includes index preparation, incident-field projection, cache
checks/copies and orchestration. The final FFTs are still computed separately
for CR and CL because their magnetic exit fields differ. Cached tree branches
that do not perform propagation again do not re-report old compute timings.

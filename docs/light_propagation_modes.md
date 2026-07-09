# Light Propagation Modes

This page explains how the simulator moves the electric field from the sample
entrance plane to the detector. It complements
[`optical_contrast_formalisms.md`](optical_contrast_formalisms.md): that page
describes the material response, while this page describes how the field is
advanced between material slices and how the final exit wave is projected to
detector pixels.

The propagation chain has three conceptually different steps:

- **Local material transmission** through each slice.
- **Optional multislice free-space propagation** between slices.
- **Final far-field propagation and flat-detector projection**.

## Coordinates And Fields

The sample-plane field is sampled on a regular grid

```text
x_j = (j - j0) dx
y_i = (i - i0) dx
```

where `dx = real_space_pixel_size`. Jones mode carries

```text
E(x, y) = [Ex(x, y), Ey(x, y)]
```

and scalar mode carries one complex field `E(x, y)` for the selected
polarization eigenmode. The vacuum wavenumber is

```text
k0 = 2 pi / lambda
```

where `lambda` is the x-ray wavelength.

## Mueller--Stokes Mode

Set `propagator_method="Stokes"` to expose polarization as Stokes vectors in
the order `[I, Q, U, V]`. The convention matches the rest of this project:
`CR = [1, -i] / sqrt(2)` maps to `[1, 0, 0, 1]`, so right-circular light has
positive `V`.

### Mueller--Stokes Theory

The Jones formalism represents a fully coherent transverse electric field by
two complex amplitudes:

```text
E = [Ex, Ey].
```

From this field one can build the Stokes vector

```text
S = [I, Q, U, V],
```

where

```text
I = |Ex|^2 + |Ey|^2
Q = |Ex|^2 - |Ey|^2
U = 2 Re(Ex Ey*)
V = 2 Im(Ex Ey*)
```

with the sign of `V` chosen as described above. `I` is total intensity, `Q`
measures horizontal-versus-vertical linear polarization, `U` measures
`+45°`-versus-`-45°` linear polarization, and `V` measures right-versus-left
circular polarization. A pure Jones state always satisfies

```text
I^2 = Q^2 + U^2 + V^2,
```

while partially polarized light satisfies

```text
I^2 >= Q^2 + U^2 + V^2.
```

The degree of polarization is therefore

```text
DoP = sqrt(Q^2 + U^2 + V^2) / I.
```

This is the first reason to use Stokes vectors: they can represent pure,
partially polarized, and unpolarized light with the same four real numbers.
For example,

```text
[1, 0, 0, +1]  pure CR
[1, 0, 0, -1]  pure CL
[1, 0, 0, +0.7]  70% CR plus 30% unpolarized light
[1, 0, 0, 0]  unpolarized light
```

Mathematically, a Stokes vector is equivalent to a `2 x 2` polarization
coherency matrix:

```text
C = 1/2 (I sigma_0 + Q sigma_1 + U sigma_2 + V sigma_3),
```

where the `sigma_i` are the Hermitian basis matrices used by the code. A
physical Stokes vector is exactly one whose coherency matrix is positive
semidefinite. This is what `validate_stokes` checks.

A Mueller matrix is a real `4 x 4` matrix that acts linearly on Stokes vectors:

```text
S_out = M S_in.
```

If the optical element is deterministic and non-depolarizing, the Mueller
matrix can be derived from a Jones matrix `J`:

```text
C_out = J C_in J†
M_ij = 1/2 Tr(sigma_i J sigma_j J†).
```

The dielectric tensor slices in the current propagation model are of this
deterministic type. In that case, Jones and Mueller--Stokes describe the same
local polarization transformation for pure inputs, while Stokes additionally
exposes the polarization state as `[I, Q, U, V]` and can carry mixed input
states through coherent-mode decomposition.

There is one important propagation subtlety. A Stokes vector describes
polarization coherence at a single transverse point, but it does not store the
complex phase relationships between different pixels. Those spatial phase
relationships are exactly what produces diffraction and holographic
interference. For this reason, the simulator does not Fourier-transform the
four Stokes components directly. Instead, it propagates one or more coherent
Jones carriers through the sample and free space, then sums their Stokes
representations incoherently. A pure state uses one carrier; a partially
polarized uniform input is diagonalized into orthogonal coherent modes of its
coherency matrix.

The returned propagator provides `input_stokes`, `exit_stokes`, and
`detector_stokes`, each with shape `(Ny, Nx, 4)`. Its hologram is the detector
plane `I` component. Utility functions in
`scattering_calculator.beam_propagator.Stokes_propagator` convert Jones
matrices to Mueller matrices, apply arbitrary Mueller matrices (including
depolarizers), convert Stokes vectors to and from coherency matrices, and
validate physical Stokes vectors.

By default the Stokes propagator uses the same pure Jones polarization as the
configured illumination. For a genuinely partially polarized incident beam,
pass a uniform physical Stokes vector through the propagator configuration:

```python
SamplePropagatorConfig(
    ...,
    propagator_method="Stokes",
    propagator_config={
        "input_stokes": [1.0, 0.0, 0.0, 0.7],  # 70% CR + 30% unpolarized
    },
)
```

The illumination image still supplies the spatial intensity envelope; the
explicit Stokes vector supplies the polarization state.

For magnetic helicity-difference holography with partially polarized beams,
compare matched helicity pairs at the same degree of polarization. For example,
to study a 70% circularly polarized beam, propagate both
`CR = [1.0, 0.0, 0.0, +0.7]` and `CL = [1.0, 0.0, 0.0, -0.7]`, then form the
hologram difference `CL - CR` before the FTH reconstruction. Subtracting an
unpolarized hologram instead changes the physical question and can hide the
helicity scaling.

A local Stokes vector contains polarization coherence but not the phase
relationship between different pixels. Directly Fourier-transforming its four
components would therefore not produce a physical diffraction pattern. The
Stokes propagator retains coherent Jones carriers internally for free-space
and Fraunhofer propagation. A pure input uses one carrier; a partially
polarized input is decomposed into orthogonal coherent modes of its coherency
matrix and the resulting Stokes fields are summed incoherently. For the
deterministic dielectric tensors currently used by the sample model, this is
the corresponding Mueller transformation at each plane. Explicit local
depolarizing elements can also be represented by the standalone
`apply_mueller` operation.

## No Multislice Propagation

When `propagate=False`, the simulator still applies the material interaction in
each layer, but it skips transverse diffraction between layers. This is the
fast thin-object style approximation.

For scalar propagation through a slice of thickness `dz` and effective complex
index `n(x, y)`,

```text
E_after(x, y) = exp(-i k0 n(x, y) dz) E_before(x, y)
```

For Jones propagation through a slice with projected dielectric tensor
`eps(x, y)`,

```text
E_after(x, y) = exp(-i k0 dz sqrt(eps(x, y))) E_before(x, y)
```

The exponential is local: each `(x, y)` pixel is updated independently. It
captures absorption, phase delay, charge contrast, XMCD, and XMLD from the
material stack, but it does not let a feature at one transverse position
diffract into its neighbors while the wave is inside the sample.

By default, the code also applies the zero-spatial-frequency free-space phase
between layers:

```text
E -> exp(-i k0 dz) E
```

This is controlled by `jones_apply_zero_order_phase` in Jones mode and
`scalar_apply_zero_order_phase` in scalar mode. It preserves the longitudinal
plane-wave phase advance while still skipping transverse FFT diffraction. Set
it to `False` only when reproducing older calculations that omitted this phase.

## Multislice Free-Space Propagation

When `propagate=True`, the simulator alternates between material transmission
and angular-spectrum free-space propagation:

```text
E_0 = incident field

for slice s:
    E_s^+ = material_transmission_s(E_s^-)
    E_{s+1}^- = P(dz_s) E_s^+        # except after the final slice
```

The free-space operator is evaluated in spatial-frequency space. With Fourier
coordinates

```text
fx = fftfreq(Nx, dx)
fy = fftfreq(Ny, dx)
kx = 2 pi fx
ky = 2 pi fy
```

the longitudinal wavevector is

```text
kz = sqrt(k0^2 - kx^2 - ky^2)
```

for propagating components. The transfer function is

```text
H(kx, ky; dz) = exp(-i kz dz)
```

and the propagation step is

```text
E(x, y; z + dz) =
    IFFT2( FFT2(E(x, y; z)) H(kx, ky; dz) )
```

Jones mode applies the same scalar free-space kernel to `Ex` and `Ey`
separately. Scalar mode applies it to the one scalar field.

For evanescent components with `kx^2 + ky^2 > k0^2`, the code does not allow
exponential growth. It uses a real decay factor

```text
alpha = sqrt(kx^2 + ky^2 - k0^2)
H_evanescent = exp(-alpha |dz|)
```

with zero real `kz` phase. This damps sub-wavelength spatial frequencies rather
than amplifying them.

## Long-Distance Vacuum Sampling

The fixed-grid transfer-function step above keeps the same transverse pixel
size before and after propagation. Li, Wojcik, and Jacobsen's nanofocusing
multislice paper states the relevant long-distance condition as: "When
subsequently propagating a wavefield in vacuum over a distance longer than
`Nt (Delta x)^2 / lambda`, sampling considerations dictate the use of an
alternative propagation approach." See the references for the full Optics
Express citation.

For a vacuum distance `dz`, the code therefore checks the multislice sampling
limit

```text
z_max = Nt dx^2 / lambda
```

where `Nt = min(Nx, Ny)`, `dx` is the transverse pixel size, and `lambda` is
the wavelength. If `|dz| > z_max`, scalar and Jones full-field propagation split
that distance into

```text
n_steps = ceil(|dz| / z_max)
```

fixed-grid substeps of length

```text
dz_step = dz / n_steps
```

before continuing. Each substep still uses

```text
E_{m+1}(x, y) =
    IFFT2( FFT2(E_m(x, y)) exp[-i kz(kx, ky) dz_step] )
```

so the regular multislice stack remains shape-preserving and the next material
slice is still sampled on the same `(x, y)` grid.

The code also exposes the paper's long-distance one-FFT Fresnel form as an
explicit propagation option:

```text
psi_{j+1}(u_x lambda z, u_y lambda z)
  = F{ psi_j(x, y) exp[-i pi (x^2 + y^2) / (lambda z)] }
    * i/(lambda z) * exp[-i pi lambda z (u_x^2 + u_y^2)]
```

through `propagate_free_space_scalar_fresnel_single_fft` and
`propagate_free_space_jones_fresnel_single_fft`. These methods return both the
propagated field and the new output sampling

```text
dy_out = lambda |z| / (Ny dx)
dx_out = lambda |z| / (Nx dx)
```

The changed sampling follows directly from the Fourier coordinates

```text
u_x = n_x / (Nx dx)
u_y = n_y / (Ny dx)
```

and the output-plane coordinates

```text
x_out = u_x lambda z
y_out = u_y lambda z
```

so adjacent output pixels are separated by `lambda |z| / (N dx)`. This is why
the regular in-stack multislice propagator does not silently switch to the
one-FFT formula: doing so would put `psi_{j+1}` on a different transverse grid
from the next dielectric slice. The explicit Fresnel methods are intended for
standalone long vacuum propagation or for optical chains where the caller
resamples the next element onto the returned grid.

## Boundary Padding And Absorbers

FFT propagation is periodic by construction. To reduce wraparound artifacts,
the propagation step can pad the field before each FFT:

```text
propagation_padding_px
propagation_padding_mode
```

The padded result is cropped back to the original sample grid after
propagation. Optional edge absorbers multiply the working field by a smooth
attenuation ramp near the boundary:

```text
propagation_absorber_width_px
propagation_absorber_strength
propagation_absorber_profile
```

When padding is enabled, the absorber is clamped to the padded margin so it
does not attenuate the returned central field.

## ROI Multislice Propagation

`multislice_propagation_roi=True` is an approximate acceleration for aperture
geometries. Instead of running the angular-spectrum FFT on the full field, the
code treats the field outside aperture regions as locally plane-wave-like and
applies only the zero-order phase:

```text
E_baseline = exp(-i k0 dz) E_in
```

For each padded aperture ROI crop `R`, the code computes a local full
propagation:

```text
E_local_R = P_R(dz) E_in|_R
correction_R = E_local_R - E_baseline|_R
```

and adds only the correction back:

```text
E_out|_R += window_R correction_R
```

The smooth `window_R` tapers the correction to zero across the ROI padding so
the pasted crop returns gradually to the plane-wave baseline. Physical aperture
supports that overlap are always merged. Padded crops that overlap are also
merged before propagation, even when
`multislice_propagation_roi_merge_overlaps=False`, to avoid adding multiple
independent corrections to the same pixels.

This is faster than full-field multislice when the holes occupy a small part of
the grid, but it is approximate because real free-space propagation is nonlocal.
Light can diffract between ROI and non-ROI pixels in the full operator.

## Final Fraunhofer Propagation

After the last material slice, the simulator forms a far-field wave by a 2-D
Fourier transform of the exit wave. For Jones mode,

```text
E_det_components(qx, qy) =
    FFTSHIFT( FFT2( IFFTSHIFT(E_exit(x, y)) ) )

I(qx, qy) = |E_det,x|^2 + |E_det,y|^2
```

For scalar mode,

```text
E_det(qx, qy) =
    FFTSHIFT( FFT2( IFFTSHIFT(E_exit(x, y)) ) )

I(qx, qy) = |E_det|^2
```

The FFT grid corresponds to spatial frequencies

```text
fx = fftshift(fftfreq(Nx, dx))
fy = fftshift(fftfreq(Ny, dx))
qx = 2 pi fx
qy = 2 pi fy
```

Equivalently, the total reciprocal-space span of the shifted FFT grid is

```text
Dq = 2 pi / dx
```

and a detector-space q-coordinate is converted to a floating FFT pixel by

```text
u_x = qx / Dq * Nx + Nx / 2
u_y = qy / Dq * Ny + Ny / 2
```

The FFT output is an ideal reciprocal-space intensity on the simulation q-grid,
not yet an image on the physical detector pixel grid.

## Far-Field Oversampling

`farfield_oversampling > 1` extends the complex exit wave before the final FFT.
It does not black-pad the sample by default. Instead, the code expects a
physical background exit wave on the larger grid, then pastes the simulated
central exit wave into it:

```text
E_large = background_exit_wave
E_large[central sample window] = E_exit
I(qx, qy) = |FFT2(E_large)|^2
```

This gives a finer reciprocal-space sampling grid while avoiding an artificial
zero boundary around the simulated sample. The detector hologram is rescaled so
the total far-field intensity matches the total intensity of the field used for
the far-field FFT.

## Projecting The Far Field To Detector Space

The physical detector is a flat plane at distance `z` from the sample. Detector
pixel coordinates are first converted from pixel indices to metres:

```text
X = (j - detector_center_x) detector_pixel_size
Y = (i - detector_center_y) detector_pixel_size
```

By default, the code uses the flat-detector angular mapping. Define

```text
r = sqrt(X^2 + Y^2)
theta = atan2(Y, X)
```

The transverse scattering-vector coordinates are

```text
qx = k0 sin(atan(r / z)) cos(theta)
qy = k0 sin(atan(r / z)) sin(theta)
```

Since

```text
sin(atan(r / z)) = r / sqrt(r^2 + z^2)
```

this is equivalent to projecting each flat-detector pixel onto its ray
direction from the sample:

```text
qx = k0 X / sqrt(X^2 + Y^2 + z^2)
qy = k0 Y / sqrt(X^2 + Y^2 + z^2)
```

If `ignore_flat_detector_curvature=True`, the code uses the small-angle linear
approximation instead:

```text
qx = k0 X / z
qy = k0 Y / z
```

The ideal FFT hologram is then sampled at the corresponding floating q-grid
coordinates with linear interpolation. In code this is equivalent to

```text
I_detector(i, j) =
    interpolate_linear(I_fft, u_y(qy), u_x(qx))
```

where

```text
u_x(qx) = qx / Dq * Nx + Nx / 2
u_y(qy) = qy / Dq * Ny + Ny / 2
```

Coordinates outside the simulated q-grid are filled with zero. Linear
interpolation is used deliberately because the hologram is an intensity; higher
order splines can ring below zero near sharp features.

## Solid-Angle Factor And Pixel Footprints

A flat detector pixel subtends a smaller solid angle as it moves away from the
optical axis. The code applies the relative flat-pixel collection factor

```text
Omega_rel(X, Y) = (z / sqrt(X^2 + Y^2 + z^2))^3
```

so the ideal detector hologram is

```text
I_detector = interpolate_linear(I_fft, qx, qy) Omega_rel(X, Y)
```

When `use_detector_pixel_footprint=False`, the projection samples each detector
pixel at its centre. When `use_detector_pixel_footprint=True`, each detector
pixel is averaged over a regular sub-sampling grid:

```text
I_pixel = (1 / M^2) sum_{a,b=1..M}
    interpolate_linear(I_fft, qx(X_a), qy(Y_b)) Omega_rel(X_a, Y_b)
```

where `M = detector_pixel_footprint_samples`. This better approximates a
finite detector-pixel area, especially when the q-grid is distorted by the flat
detector geometry.

## Detector-Limited Sample Resolution

The detector layout computes the real-space resolution associated with the
available q-span as

```text
resolution = 2 pi / (max(qx) - min(qx))
```

The sweep script then uses

```text
real_space_pixel_size = resolution / oversampling
sample_shape = oversampling * detector_shape
```

for the simulation grid. This means `oversampling` controls how finely the
sample is sampled relative to the detector-limited real-space resolution,
whereas `farfield_oversampling` controls the q-grid sampling of the final FFT.

## Choosing A Propagation Mode

Use **no multislice propagation** (`propagate=False`) when the sample is thin
enough that transverse diffraction inside the stack is negligible, or when you
want the fastest local-transmission approximation.

Use **full-field multislice propagation** (`propagate=True`,
`multislice_propagation_roi=False`) when diffraction between all sample pixels
matters. This is the most conservative propagation mode.

Use **ROI multislice propagation** (`propagate=True`,
`multislice_propagation_roi=True`) when the expensive diffractive structure is
localized near aperture holes and you have validated that the ROI approximation
is sufficient for the geometry.

The final Fraunhofer FFT and detector q-space projection are always used to
turn the exit wave into an ideal detector hologram.

## References

Kenan Li, Michael Wojcik, and Chris Jacobsen, "Multislice does it
all--calculating the performance of nanofocusing X-ray optics," *Optics
Express* **25**(3), 1831 (2017). DOI:
[`10.1364/OE.25.001831`](https://doi.org/10.1364/OE.25.001831). Optica page:
[`oe-25-3-1831`](https://opg.optica.org/abstract.cfm?URI=oe-25-3-1831).

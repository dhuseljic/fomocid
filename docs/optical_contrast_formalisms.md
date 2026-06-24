# Optical Contrast Formalisms

This page collects the light-matter interaction models used by the coherent
scattering simulator. The goal is to make explicit what is meant by charge
contrast, XMCD contrast, XMLD contrast, scalar propagation, Jones propagation,
and vector contrast with a local light-momentum direction.

For the separate question of how the field is advanced between slices and then
projected to detector pixels, see
[`light_propagation_modes.md`](light_propagation_modes.md).

The implementation has two main optical formalisms:

- **Scalar refractive-index propagation**: one complex field is propagated for
  the selected polarization eigenmode. It is fast and uses an effective scalar
  refractive index per voxel.
- **Jones dielectric-tensor propagation**: a two-component transverse electric
  field is propagated through a 2-by-2 dielectric tensor per voxel. This keeps
  polarization mixing and is the reference formalism for vector contrast.

Both formalisms use the same material database convention. For every material
and photon energy, the simulator stores

```text
n = [n0, dn_c, dn_l]
```

where `n0` is the isotropic complex refractive index, `dn_c` is the circular
magneto-optic contribution, and `dn_l` is the linear magneto-optic
contribution. The complex convention is the usual x-ray one: the real part
controls phase advance/refraction and the imaginary part controls absorption.

## Charge, XMCD, And XMLD Channels

For a unit magnetization vector

```text
m = (mx, my, mz)
```

the three optical channels are:

- **Charge/isotropic contrast**: independent of magnetization. In scalar form
  it is `n0`; in tensor form it is `eps0 I`.
- **XMCD/circular contrast**: odd in magnetization and helicity. It couples to
  the longitudinal magnetization component seen by the light, so the physical
  vector form is proportional to `m . k`, where `k` is the local propagation
  direction.
- **XMLD/linear contrast**: even in magnetization reversal. It depends on the
  projection of the magnetization into the transverse polarization plane, so
  reversing `m -> -m` does not change the XMLD term.

The Jones tensor uses the small-anisotropy dielectric increments

```text
eps0 = n0^2
eps_c = 2 n0 dn_c
eps_l = 2 n0 dn_l
```

The scalar formalism uses the refractive-index increments `dn_c` and `dn_l`
directly.

## Scalar Refractive-Index Formalism

Scalar propagation replaces the full Jones interaction by one effective
refractive index for the chosen input polarization. The code first computes
polarization coefficients from the Jones vector `p`:

```text
c_pol = p^dagger C p
l_pol = p^dagger L p

C = [[0,  i],
     [-i, 0]]

L = [[1,  0],
     [0, -1]]
```

The shipped polarization vectors are

```text
CR = [1, -i] / sqrt(2)
CL = [1,  i] / sqrt(2)
LH = [1,  0]
LV = [0,  1]
```

so the coefficients are

| Polarization | `c_pol` | `l_pol` |
|---|---:|---:|
| `CR` | `+1` | `0` |
| `CL` | `-1` | `0` |
| `LH` / `x` | `0` | `+1` |
| `LV` / `y` | `0` | `-1` |

For an arbitrary linear angle `theta`, the current helper returns
`p = [sin(theta), cos(theta)]`, so `c_pol = 0` and
`l_pol = sin(theta)^2 - cos(theta)^2`.

For a material pixel, the scalar effective index is

```text
n_eff = n0
      + c_pol mz dn_c
      + l_pol (|mx|^2 - |my|^2) dn_l
```

The terms have the following interpretation:

- `n0` is charge contrast.
- `c_pol mz dn_c` is scalar XMCD for normal-incidence propagation along `z`.
  It changes sign when the helicity changes from `CR` to `CL`, and it changes
  sign when `mz` changes sign.
- `l_pol (|mx|^2 - |my|^2) dn_l` is scalar XMLD for the lab `x/y`
  polarization basis. It is even under `m -> -m`.

The aperture mask mixes material with vacuum:

```text
n_voxel = mask n_eff + (1 - mask)
```

where `mask = 1` is material and `mask = 0` is vacuum. For each slice of
physical thickness `dz`, the scalar material interaction is applied as

```text
E_out(x, y) = exp(-i k0 n_voxel(x, y) dz) E_in(x, y)
k0 = 2 pi / lambda
```

Optional multislice propagation then applies the same angular-spectrum
free-space operator used by Jones mode:

```text
H(kx, ky; dz) = exp(-i kz dz)
kz = sqrt(k0^2 - kx^2 - ky^2)
```

The scalar formalism is exact relative to Jones when the chosen polarization is
a local eigenvector of the Jones interaction, for example circular polarization
through a purely circular normal-incidence response. It is not the right model
when the sample rotates or mixes polarization components that should be kept in
the output field.

## Jones Dielectric-Tensor Formalism

Jones mode propagates the transverse electric field

```text
E = [Ex, Ey]
```

through a projected 2-by-2 dielectric tensor. For normal incidence with the
lab transverse basis `e1 = x`, `e2 = y`, and `k = z`, the material tensor can be
written as

```text
eps_11 = eps0 + eps_l (|mx|^2 - |my|^2)
eps_22 = eps0 - eps_l (|mx|^2 - |my|^2)
eps_12 =  i eps_c mz + 2 eps_l mx my
eps_21 = -i eps_c mz + 2 eps_l mx my
```

The diagonal isotropic part is charge contrast. The antisymmetric imaginary
off-diagonal pair, `+i eps_c mz` and `-i eps_c mz`, is XMCD. The symmetric
diagonal/off-diagonal linear terms are XMLD.

For a slice of thickness `dz`, Jones mode evaluates the matrix function

```text
E_out = exp(-i k0 dz sqrt(eps)) E_in
```

pixel by pixel. In the implementation, diagonal pixels are handled with the
fast scalar diagonal exponentials. Mixed pixels are evaluated by the analytic
2-by-2 matrix function using the two eigenvalues of `eps`, so the code does not
need to form a dense matrix exponential for every pixel.

The mask/vacuum mixture is performed at the dielectric level. Material
fractions contribute their material tensor and vacuum contributes the identity
response, because vacuum has `eps = 1`.

## Vector Contrast And Tensor Projection

The normal-incidence equations above assume that the light momentum is the lab
`z` direction. That is too restrictive for tilted beams, tilted samples, and
multislice propagation through apertures. The vector formalism keeps the
magneto-optic response in lab 3-D coordinates and projects it onto the local
transverse Jones basis.

For a propagation direction `k`, choose two orthonormal transverse basis
vectors `e1` and `e2` such that

```text
e1 . k = 0
e2 . k = 0
e1 x e2 = k
```

The simulator stores the direction-independent response as

```text
g = eps_c m
Q = eps_l m m^T
```

where `g` is the gyrotropic XMCD vector and `Q` is the XMLD tensor. The
projected quantities are

```text
g_k  = g . k
q_ij = e_i . Q . e_j
```

and the projected Jones tensor is

```text
eps_11 = eps_iso + q_11 - q_22
eps_22 = eps_iso - q_11 + q_22
eps_12 =  i g_k + 2 q_12
eps_21 = -i g_k + 2 q_12
```

This is the central vector-contrast equation. The circular term is not tied to
`mz`; it is tied to `m . k`. If `k = z`, then `g_k = eps_c mz` and the equation
reduces to the normal-incidence Jones tensor. If the beam is tilted, an
in-plane magnetization component can contribute to XMCD through its projection
onto `k`.

## Fixed Beam-Direction Projection

For tilted illumination without local-k projection, the tensor stack is
projected once using the nominal beam direction. The tilt convention is
`alpha_beam = (alpha_y, alpha_x)`, and the beam direction is

```text
k_beam = normalize([tan(alpha_x), tan(alpha_y), 1])
```

This mode captures the geometric vector contrast for a tilted incident beam but
keeps the tensor stack fixed during propagation. It is appropriate when the
wavefront direction is well approximated by the incident direction throughout
the sample.

## Local Momentum Projection

With `dielectric_tensor_local_k_projection=True`, Jones mode computes the
projection direction from the evolving electric field before every material
slice. For a Jones field `E_p(x, y)`, where `p` runs over the two transverse
components, the code estimates the intensity-weighted phase gradient:

```text
I = sum_p |E_p|^2

grad_x phi = Im(sum_p conj(E_p) dE_p/dx) / I
grad_y phi = Im(sum_p conj(E_p) dE_p/dy) / I
```

The simulator phase convention gives a positive x tilt the phase
`exp(-i k0 x sin(alpha_x))`, so the normalized local direction is

```text
kx / k0 = -grad_x phi / k0
ky / k0 = -grad_y phi / k0
kz / k0 = sqrt(1 - (kx / k0)^2 - (ky / k0)^2)
```

If the transverse magnitude is numerically too large, it is clipped just below
one so the projected direction remains propagating and `kz` stays real. The
code then rebuilds `e1`, `e2`, and the projected tensor for that slice.

This local-k method is the most physical vector-contrast option currently in
the code. It lets diffraction, focusing, aperture steering, and earlier
material slices change the local `m . k` contrast seen by later slices. The
cost is that the projected tensor must be rebuilt slice by slice, so compact
Jones ROI tensor storage is disabled for this path.

## Choosing A Method

Use this rule of thumb:

- Use **scalar propagation** for fast sweeps when the chosen polarization is a
  local eigenmode and polarization mixing is not important.
- Use **Jones propagation** when XMLD, arbitrary linear polarization,
  polarization rotation, or mixed transverse components matter.
- Use **fixed beam-direction vector contrast** when a tilted incident beam
  changes the XMCD projection but the local wavefront direction is otherwise
  well represented by the incident direction.
- Use **local-k vector contrast** when multislice diffraction or strong
  aperture structure should change the local light momentum inside the sample.

The tilted multilayer demonstration notebook,
[`tutorial_tilted_magnetic_layer_multislice.ipynb`](../tutorials/tutorial_tilted_magnetic_layer_multislice.ipynb),
is the focused runnable example for fixed beam-direction and local-k vector
contrast.

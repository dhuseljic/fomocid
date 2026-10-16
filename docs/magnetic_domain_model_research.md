# A controlled, fast magnetic-domain model

A first experimental implementation is now available as
`create_constrained_domain_pattern`; see [implementation and trial measurements](constrained_magnetic_domains.md).
This document describes its literature basis and remaining validation work.
The procedural `domain_pattern` generator does not implement constrained micromagnetics.
The current generator and catalogue are useful procedural baselines. They lack
an explicit dipolar interaction, directional stripe-order control, and a
calibrated relation from requested length to the measured domain-size distribution.

## Relevant primary literature

1. E. A. Jagla, *Numerical simulations of two-dimensional magnetic domain
   patterns*, Physical Review E **70**, 046204 (2004).
   [Publisher](https://doi.org/10.1103/PhysRevE.70.046204),
   [open full text](https://arxiv.org/html/cond-mat/0402406).
   A scalar double-well field, interfacial stiffness and dipolar repulsion
   reproduce stripes, bubbles and labyrinthine/polygonal morphologies.
   The dipolar convolution is evaluated by FFT. Field history and metastability
   matter. This is a particularly relevant starting point for a fast morphology
   backend, rather than independently drawing prescribed objects.
2. E. A. Jagla, *Hysteresis loops of magnetic thin films with perpendicular
   anisotropy*, Physical Review B **72**, 094406 (2005).
   [Publisher](https://doi.org/10.1103/PhysRevB.72.094406),
   [open full text](https://arxiv.org/pdf/cond-mat/0412461).
   Reduces perpendicular-anisotropy magnetization dynamics to a scalar model
   with exchange, dipolar interactions and structural disorder, and compares
   spatial patterns and hysteresis with Co/Pt multilayers. Its assumptions
   concern predominantly perpendicular domains with relatively narrow walls;
   it does not supply full vector wall dynamics.
3. C. B. Muratov, *Theory of domain patterns in systems with long-range
   interactions of Coulomb type*, Physical Review E **66**, 066108 (2002).
   [Publisher](https://doi.org/10.1103/PhysRevE.66.066108),
   [preprint](https://arxiv.org/abs/cond-mat/0209570).
   Explains how competing interfacial and nonlocal energies generate spots,
   lamellae and hexagonal structures, with characteristic scales and shape
   instabilities. Coulomb and magnetic dipolar kernels are different: this is
   useful pattern-selection theory, not an interchangeable magnetic kernel.
4. *Phase prediction method for pattern formation in time-dependent
   Ginzburg-Landau dynamics for kinetic Ising model without a priori assumptions
   of domain patterns*, Physical Review B **103**, 094408 (2021).
   [Publisher](https://doi.org/10.1103/PhysRevB.103.094408).
   Studies material-parameter dependence of patterns in a two-dimensional
   kinetic Ising/TDGL model with dipolar interactions and finite film thickness.
   Relevant to building a measured phase catalogue instead of assigning
   morphology names from seed choices alone.

## Recommended model and controls

Use a two-dimensional scalar phase field with local double-well energy,
interfacial stiffness, a thin-film dipolar kernel, optional directional stiffness,
and quenched disorder. Compute spatial derivatives and nonlocal convolution
spectrally. This recommendation combines the literature's reduced physical
model with **our additional fixed-area generation constraint**. It is not a
verbatim implementation of one of these papers.

The smooth internal field is an auxiliary variable for stable wall motion;
the requested output is a domain mask with m_z in {-1,+1}. Keeping a smooth
auxiliary field avoids forcing the evolving interfaces onto a one-pixel lattice.

Public controls should describe observable targets rather than raw solver
parameters:

| Proposed control | Meaning | How it must be enforced or verified |
|---|---|---|
| `target_magnetization` | Normalized net M_z/M_s, in [-1,1] | For fully magnetized domains f_up=(1+M)/2. Constrain the **sign-area fraction**, not the mean of an arbitrary continuous field. |
| `characteristic_length` | Preferred domain repeat/spacing in physical units | Tune the interaction spectrum and verify the observed spectral peak; state the finite-size/resolution tolerance. |
| `alignment` and `direction` | Preferred stripe-axis ordering | Directional energetic bias, separate from a seeded initial orientation; measure nematic order. |
| `disorder` and `disorder_length` | Pinning strength and correlation scale | Quenched spatial disorder; measure width distributions, defect density and roughness. |
| `initial_order` and `initial_state` | Preparation/history, including random, stripe, bubble seeds | Affects metastable outcomes; do not promise final order from the initial seed alone. |
| `relaxation` | Coarsening/annealing effort | Stop using recorded convergence criteria; distinguish partially relaxed and converged outputs. |
| `seed` | Reproducible stochastic realization | Record every source of randomness. |

`target_magnetization` is a synthetic fixed-magnetization constraint, not a
specified applied field. In a physical field-controlled model the resulting
magnetization is an observable and depends on history. Expose those as separate
operating modes, without silently treating them as equivalent.

The fixed-area constraint should act during relaxation so that domain shapes
adapt to the requested occupancy. A final quantile adjustment alone can produce
the right area while substantially changing wall positions and connectivity.
A constraint on the continuous-field mean alone also fails to guarantee the
requested binary occupancy. The implementation must explicitly account for
this distinction and report the achieved output constraint.

## Length and shape cannot all be independent

For straight stripes of repeat L and positive area fraction f, the widths are
w_up=f L and w_down=(1-f)L. Both widths cannot be prescribed independently
along with M and L. Similarly, for separated approximately circular bubbles,
the minority fraction links bubble density and radius. Distinguish repeat
spacing, stripe thickness and bubble diameter in both the API and measurements.
A single nominal length should have a single documented meaning.

Certain requested combinations will be incompatible or metastable. The solver
should report measured outcomes and tolerances rather than label every output
as the requested phase. At M=+1 or -1 the output is uniform and no domain period
can be measured.

## Validation before calling this a micromagnetic surrogate

Compare representative outputs against reference micromagnetic simulations
at matched sampling, geometry and normalized magnetization. Use the structure
factor, two-point correlation, phase fractions, wall length, component area and
shape distributions, stripe nematic order and bubble-center sixfold order.
Repeat across seeds, grid resolutions and histories. Measure runtime alongside
those errors. A scalar morphology approximation is valuable for generating
scattering datasets even when it does not reproduce vector chirality, DMI,
precession, switching trajectories or skyrmion topology.

Keep procedural generation and physical relaxation available as distinct
backends. Preserve the current catalogue as procedural evidence; it is not yet
a material-calibrated magnetic phase diagram.

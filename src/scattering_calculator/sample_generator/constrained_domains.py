"""Experimental fixed-area scalar thin-film domain relaxation.

Inspired by Jagla, PRE 70, 046204 (2004), doi:10.1103/PhysRevE.70.046204.
The fixed sign-area projection, directional penalty and interface are our
extensions; this is not an implementation of vector LLG dynamics.
"""
from time import perf_counter
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from .domain_catalogue import measure_domain_pattern


def _relax_constrained_domains(
    sz_array, characteristic_length, *, target_magnetization=0., alignment=0.,
    direction=0., disorder=0., disorder_length=1., initial_state='random',
    initial_order=0., initial_domain_count=1, worm_length=None, worm_waviness=None,
    wall_stiffness=2., n_steps=300, time_step=.2,
    convergence_tolerance=1e-5, seed=None, real_space_pixel_size=1., **options,
):
    """Relax a scalar field with dipolar competition and fixed binary occupancy.

    Direct lengths are pixels; MagneticPatternConfig accepts metres. The target
    normalized net magnetization sets the number of +1 output pixels, rounded
    to the closest integer. A sign-area projection acts after *every* evolution
    step. Returned magnetization is always +/-1, independent of the auxiliary
    continuous field. characteristic_length is a preferred full repeat/spacing,
    not a prescribed stripe thickness or bubble diameter; measured values are
    returned. direction is the stripe-axis angle from +x towards +y (radians).

    The Fourier interaction is beta*k^2-gamma*|k|, with its minimum at
    2*pi/characteristic_length, plus an alignment penalty for wavevectors along
    the stripe axis. This is the long-wavelength dipolar approximation, not a
    finite-thickness demagnetizing kernel. Local double-well relaxation uses a
    stabilized semi-implicit step. Projection and clipping mean energy descent
    is not guaranteed; the measured trace is provided rather than assumed.
    Boundary conditions are periodic. Disorder is a quenched random field.
    """
    options.pop('coordinate_offset', None)
    if options:
        raise TypeError(f'Unknown constrained-domain controls: {sorted(options)}')
    shape = tuple(int(v) for v in sz_array)
    if len(shape) != 2 or min(shape) < 8 or any(int(v) != v for v in sz_array):
        raise ValueError('sz_array must contain two integer dimensions of at least 8')
    controls = dict(characteristic_length=characteristic_length,
                    target_magnetization=target_magnetization, alignment=alignment,
                    direction=direction, disorder=disorder, disorder_length=disorder_length,
                    initial_order=initial_order, wall_stiffness=wall_stiffness,
                    time_step=time_step, convergence_tolerance=convergence_tolerance)
    if any(not np.isscalar(v) or not np.isfinite(v) for v in controls.values()):
        raise ValueError('All constrained-domain controls must be finite scalars')
    if not -1 <= target_magnetization <= 1 or not 0 <= initial_order <= 1:
        raise ValueError('target_magnetization must be in [-1,1]; initial_order in [0,1]')
    if characteristic_length < 4 or alignment < 0 or disorder < 0 or disorder_length < 0:
        raise ValueError('characteristic_length must be >=4 pixels; alignment and disorder controls nonnegative')
    if wall_stiffness < 1 or time_step <= 0 or convergence_tolerance <= 0:
        raise ValueError('wall_stiffness must be >=1; time_step and convergence_tolerance positive')
    if int(n_steps) != n_steps or n_steps < 1:
        raise ValueError('n_steps must be a positive integer')
    if initial_state not in ('random', 'stripes', 'hexagonal', 'worms'):
        raise ValueError('initial_state must be random, stripes, hexagonal or worms')
    if int(initial_domain_count) != initial_domain_count or initial_domain_count < 1:
        raise ValueError('initial_domain_count must be a positive integer')
    if worm_length is not None and (not np.isfinite(worm_length) or worm_length <= 0):
        raise ValueError('worm_length must be positive or None')
    if worm_waviness is not None and (not np.isfinite(worm_waviness) or worm_waviness < 0):
        raise ValueError('worm_waviness must be nonnegative or None')
    start = perf_counter()
    rng = np.random.default_rng(seed)
    q0 = 2*np.pi/characteristic_length
    ky = 2*np.pi*np.fft.fftfreq(shape[0])[:, None]
    kx = 2*np.pi*np.fft.fftfreq(shape[1])[None, :]
    k = np.hypot(kx, ky)
    parallel = kx*np.cos(direction)+ky*np.sin(direction)
    interaction = (k/q0)**2 - 2*k/q0 + alignment*(parallel/q0)**2
    interaction[0,0] = 0.
    alpha = float(wall_stiffness)
    denominator = 1 + time_step*(2*alpha + interaction)
    count = int(np.rint((target_magnetization+1)*.5*np.prod(shape)))
    tie_breaker = rng.uniform(-1e-12, 1e-12, size=shape)

    def project(field):
        """Shift the level set to the requested sign area, retaining smooth walls."""
        if count == 0: return -np.ones(shape)
        if count == field.size: return np.ones(shape)
        values = field + tie_breaker
        split = field.size-count
        partition = np.partition(values.ravel(), (split-1, split))
        threshold = .5*(partition[split-1]+partition[split])
        return np.clip(values-threshold, -1., 1.)

    random = rng.standard_normal(shape)
    shell = np.exp(-.5*((k/q0-1)/.25)**2)
    random = np.fft.ifft2(np.fft.fft2(random)*shell).real
    random /= max(float(random.std()), 1e-12)
    yy,xx = np.indices(shape)
    phases = rng.uniform(0,2*np.pi,2)
    ordered = np.zeros(shape)
    if initial_state == 'stripes':
        ordered = np.cos(q0*(-xx*np.sin(direction)+yy*np.cos(direction))+phases[0])
    elif initial_state == 'hexagonal':
        for i in range(3):
            angle=direction+2*np.pi*i/3
            phase=phases[i] if i<2 else -phases.sum()
            ordered += np.cos(q0*(xx*np.cos(angle)+yy*np.sin(angle))+phase)
    elif initial_state == 'worms':
        # Finite, curved minority-domain seeds, not stripes wrapping the box.
        # Curve distance sets the initial geometry; area projection sets width.
        length = .6*min(shape) if worm_length is None else float(worm_length)
        waviness = .06*min(shape) if worm_waviness is None else float(worm_waviness)
        curves = []
        along = np.linspace(-length/2, length/2, max(64, int(2*length)))
        axis = np.array([np.sin(direction), np.cos(direction)])
        normal = np.array([np.cos(direction), -np.sin(direction)])
        spacing = .6*min(shape)/max(1,initial_domain_count)
        for index in range(int(initial_domain_count)):
            offset = (index-(initial_domain_count-1)/2)*spacing
            center = .5*np.array(shape)+offset*normal
            phase = rng.uniform(0,2*np.pi)
            bend = waviness*np.sin(2*np.pi*along/length+phase)
            curves.append((center[None,:]+along[:,None]*axis+bend[:,None]*normal)%np.array(shape))
        tree=cKDTree(np.concatenate(curves),boxsize=shape)
        distance=np.empty(np.prod(shape))
        for start_pixel in range(0,distance.size,65536):
            stop_pixel=min(distance.size,start_pixel+65536)
            rows,cols=np.divmod(np.arange(start_pixel,stop_pixel),shape[1])
            distance[start_pixel:stop_pixel]=tree.query(np.column_stack((rows,cols)))[0]
        # Minority polarity reverses with target M; amplitudes remain +/-1.
        ordered = -distance.reshape(shape)
        if target_magnetization >= 0: ordered = -ordered
    if initial_state != 'random':
        ordered /= max(float(ordered.std()),1e-12)
    blend = initial_order if initial_state != "random" else 0.
    field = project(.5*((1-blend)*random+blend*ordered))
    pinning = gaussian_filter(rng.standard_normal(shape),disorder_length, mode='wrap')
    pinning *= disorder/max(float(pinning.std()),1e-12)

    def energy(u):
        spectrum=np.fft.fft2(u)
        return float(alpha*np.mean((u*u-1)**2)/4
                     + .5*np.sum(interaction*abs(spectrum)**2)/u.size**2
                     - np.mean(pinning*u))

    trace=[]
    converged = count in (0,field.size)
    residual = 0.
    steps = 0
    stable_checks = 0
    if not converged:
        for step in range(int(n_steps)):
            previous = field
            rhs = field + time_step*(3*alpha*field-alpha*field**3+pinning)
            field = project(np.fft.ifft2(np.fft.fft2(rhs)/denominator).real)
            residual = float(np.sqrt(np.mean((field-previous)**2)))
            steps = step+1
            stable_checks = stable_checks+1 if residual < convergence_tolerance else 0
            if steps == 1 or steps % 10 == 0 or stable_checks >= 5:
                trace.append(dict(step=steps, energy=energy(field), residual=residual,
                                  up_pixels=int(np.count_nonzero(field>0))))
            if stable_checks >= 5:
                converged=True
                break
    pattern = np.where(field>0,1.,-1.)
    # Always include the actual last iterate in the provenance trace.
    if not trace or trace[-1]['step'] != steps:
        trace.append(dict(step=steps,energy=energy(field),residual=residual,
                          up_pixels=int(np.count_nonzero(field>0))))
    spectrum=abs(np.fft.fft2(pattern-pattern.mean()))**2
    weights=spectrum.sum()
    # Stripe orientation order, sign-independent and zero for isotropic spectra.
    angle=np.arctan2(ky,kx)
    nematic=float(abs(np.sum(spectrum*np.exp(2j*angle)))/weights) if weights>0 else None
    metrics=measure_domain_pattern(pattern)
    # Average over annuli so one unusually strong Fourier pixel does not
    # determine the size estimate of an otherwise isotropic domain texture.
    fundamental=2*np.pi/max(shape)
    bins=np.rint(k/fundamental).astype(int)
    totals=np.bincount(bins.ravel(),weights=spectrum.ravel())
    populations=np.bincount(bins.ravel())
    radial=totals/np.maximum(populations,1)
    radial[0]=0
    peak=int(np.argmax(radial))
    delta=0.
    if 0 < peak < len(radial)-1:
        curvature=radial[peak-1]-2*radial[peak]+radial[peak+1]
        if curvature < 0:
            delta=float(np.clip(.5*(radial[peak-1]-radial[peak+1])/curvature,-.5,.5))
    metrics['radial_period_px']=float(2*np.pi/((peak+delta)*fundamental)) if peak>0 else None
    metadata=dict(generator='constrained_dipolar_domains', model='scalar_longwave_dipolar',
                  reference='doi:10.1103/PhysRevE.70.046204',
                  **controls, n_steps_requested=int(n_steps),steps_completed=steps,
                  converged=converged,residual=residual,trace=trace,
                  target_up_pixels=count, up_fraction_measured=float(np.mean(pattern>0)),
                  mean_mz=float(pattern.mean()),nematic_order=nematic,
                  measurements=metrics,seed=seed,elapsed_seconds=perf_counter()-start,
                  initial_state=initial_state,initial_domain_count=int(initial_domain_count),
                  worm_length=worm_length,worm_waviness=worm_waviness,
                  initial_worm_length_px=length if initial_state == "worms" else None,
                  initial_worm_waviness_px=waviness if initial_state == "worms" else None,
                  boundary_conditions='periodic')
    return pattern,metadata


def create_constrained_domain_pattern(
    sz_array, characteristic_length, *, length_tolerance=.15,
    max_length_iterations=5, **controls,
):
    """Generate fixed-area +/-1 domains with a measured length target.

    Main controls: target_magnetization [-1,1], alignment >=0, direction in
    radians, disorder >=0, disorder_length in pixels, initial_state ('random',
    'stripes', 'hexagonal', 'worms'), initial_order [0,1], initial_domain_count,
    worm_length/worm_waviness (preparation lengths), wall_stiffness >=1, n_steps,
    time_step, convergence_tolerance, and seed. Pipeline lengths are metres.

    characteristic_length is the target dominant spectral repeat, not bubble
    diameter. At most max_length_iterations same-seed relaxations adjust the
    preferred interaction scale using the measured output. The best result is
    returned with length_target_met and convergence metadata; incompatible or
    unresolved targets are not silently claimed to be satisfied. Use one
    iteration to disable feedback. For saturated outputs the length is undefined.
    The mask area constraint acts at every relaxation step, on the full grid.
    See _relax_constrained_domains for the model and its limitations.
    """
    if not np.isfinite(length_tolerance) or length_tolerance <= 0:
        raise ValueError('length_tolerance must be finite and positive')
    if int(max_length_iterations) != max_length_iterations or max_length_iterations < 1:
        raise ValueError('max_length_iterations must be a positive integer')
    # Hold the same realization across calibration attempts, including seed=None.
    controls = dict(controls)
    if controls.get('seed') is None:
        controls['seed'] = int(np.random.default_rng().integers(0,2**31-1))
    start=perf_counter()
    requested=float(characteristic_length)
    preferred=requested
    history=[]
    best=None
    for attempt in range(int(max_length_iterations)):
        pattern,meta=_relax_constrained_domains(sz_array,preferred,**controls)
        measured=meta['measurements']['radial_period_px']
        error=abs(measured/requested-1) if measured is not None else float('inf')
        history.append(dict(attempt=attempt+1, preferred_length_px=preferred,
                            measured_length_px=measured, relative_error=error if measured is not None else None,
                            steps_completed=meta['steps_completed'],converged=meta['converged']))
        if best is None or error < best[0]: best=(error,pattern,meta,preferred)
        if measured is None or error <= length_tolerance: break
        lower = [item for item in history if item['measured_length_px'] is not None
                 and item['measured_length_px'] < requested]
        upper = [item for item in history if item['measured_length_px'] is not None
                 and item['measured_length_px'] > requested]
        if lower and upper:
            lo,hi=min(((a,b) for a in lower for b in upper),
                      key=lambda pair:abs(pair[0]['preferred_length_px']-pair[1]['preferred_length_px']))
            preferred=.5*(lo['preferred_length_px']+hi['preferred_length_px'])
        else:
            preferred=max(4.,preferred*requested/measured)
    error,pattern,meta,preferred=best
    measured=meta['measurements']['radial_period_px']
    meta.update(characteristic_length=requested,effective_preferred_length_px=preferred,
                measured_length_px=measured,length_relative_error=error if measured is not None else None,
                length_target_met=bool(measured is not None and error <= length_tolerance),
                length_tolerance=float(length_tolerance),calibration_history=history,
                elapsed_seconds=perf_counter()-start)
    return pattern,meta

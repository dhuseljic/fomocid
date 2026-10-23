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
    convergence_tolerance=1e-5, seed=None, real_space_pixel_size=1., progress_callback=None, measure_components=True, **options,
):
    """Relax a scalar field with dipolar competition and optional fixed occupancy.

    Direct lengths are pixels; MagneticPatternConfig accepts metres. The target
    normalized net magnetization sets the number of +1 output pixels, rounded
    to the closest integer. A sign-area projection acts after *every* evolution
    step when a numeric target is supplied. With target_magnetization=None,
    no level-set shift or fixed-area projection is applied: the model evolves
    its own mean and sign fractions. Returned magnetization is always +/-1, independent of the auxiliary
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
    if any(not np.isscalar(v) or not np.isfinite(v) for key, v in controls.items()
           if not (key == 'target_magnetization' and v is None)):
        raise ValueError('All constrained-domain controls must be finite scalars')
    if (target_magnetization is not None and not -1 <= target_magnetization <= 1) or not 0 <= initial_order <= 1:
        raise ValueError('target_magnetization must be None or in [-1,1]; initial_order in [0,1]')
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
    if not isinstance(measure_components, (bool, np.bool_)):
        raise ValueError('measure_components must be boolean')
    start = perf_counter()
    timings = dict(initialization=0., local_update=0., spectral_update=0.,
                   area_projection=0., convergence_check=0., trace=0.,
                   relaxation=0., measurements=0., total=0.)
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
    # Taking real(ifft2(...)) symmetrizes the full filter. Preserve that
    # behavior at Nyquist rows/columns for rotated anisotropic interactions.
    half_width = shape[1] // 2 + 1
    mirror_y = (-np.arange(shape[0])) % shape[0]
    mirror_x = (-np.arange(half_width)) % shape[1]
    real_filter = .5 * (1 / denominator[:, :half_width] +
                       1 / denominator[np.ix_(mirror_y, mirror_x)])
    count = None if target_magnetization is None else int(np.rint((target_magnetization+1)*.5*np.prod(shape)))
    tie_breaker = None if count is None else rng.uniform(-1e-12, 1e-12, size=shape)

    def project(field):
        """Shift the level set to the requested sign area, retaining smooth walls."""
        if count is None: return np.clip(field, -1., 1.)
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
        if target_magnetization is None:
            # Set an initial tube width, without projecting its final occupancy.
            ordered += .25 * characteristic_length
        elif target_magnetization >= 0:
            ordered = -ordered
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

    timings['initialization'] = perf_counter() - start
    if progress_callback is not None:
        progress_callback(dict(stage='initialization', elapsed_seconds=timings['initialization']))
    trace=[]
    converged = count in (0,field.size)
    residual = 0.
    steps = 0
    stable_checks = 0
    relaxation_start = perf_counter()
    last_reported_step = -1
    if not converged:
        for step in range(int(n_steps)):
            previous = field
            tick = perf_counter()
            cube = field * field * field
            rhs = field + time_step*(3*alpha*field-alpha*cube+pinning)
            now = perf_counter(); timings['local_update'] += now - tick; tick = now
            updated = np.fft.irfft2(np.fft.rfft2(rhs) * real_filter, s=shape)
            now = perf_counter(); timings['spectral_update'] += now - tick; tick = now
            field = project(updated)
            now = perf_counter(); timings['area_projection'] += now - tick; tick = now
            residual = float(np.sqrt(np.mean((field-previous)**2)))
            now = perf_counter(); timings['convergence_check'] += now - tick
            steps = step+1
            stable_checks = stable_checks+1 if residual < convergence_tolerance else 0
            if steps == 1 or steps % 10 == 0 or stable_checks >= 5:
                tick = perf_counter()
                trace.append(dict(step=steps, energy=energy(field), residual=residual,
                                  up_pixels=int(np.count_nonzero(field>0))))
                timings['trace'] += perf_counter() - tick
            if progress_callback is not None and (steps == 1 or steps % 50 == 0 or stable_checks >= 5):
                progress_callback(dict(stage='relaxation', step=steps, max_steps=int(n_steps),
                    elapsed_seconds=perf_counter()-relaxation_start, residual=residual))
                last_reported_step = steps
            if stable_checks >= 5:
                converged=True
                break
    timings['relaxation'] = perf_counter() - relaxation_start
    if progress_callback is not None and last_reported_step != steps:
        progress_callback(dict(stage='relaxation', step=steps, max_steps=int(n_steps),
            elapsed_seconds=timings['relaxation'], residual=residual))
    measurement_start = perf_counter()
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
    # Report the displayed open field; periodic analysis remains available explicitly.
    if measure_components:
        metrics=measure_domain_pattern(pattern, periodic=False, boundary_context_periodic=True)
    else:
        # Production classifies the visible periodic domains once, after the OH
        # is known. Calibration needs only the spectral length, not components.
        peak_y, peak_x = np.unravel_index(np.argmax(spectrum), spectrum.shape)
        peak_frequency = k[peak_y, peak_x]
        metrics = dict(up_fraction=float(np.mean(pattern > 0)), mean_mz=float(pattern.mean()),
            dominant_period_px=float(2*np.pi/peak_frequency) if weights > 0 and peak_frequency > 0 else None)
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
    timings['measurements'] = perf_counter() - measurement_start
    timings['total'] = perf_counter() - start
    if progress_callback is not None:
        progress_callback(dict(stage='measurements', elapsed_seconds=timings['measurements']))
    metadata=dict(timings_seconds=timings, component_diagnostics=bool(measure_components), generator='constrained_dipolar_domains', model='scalar_longwave_dipolar',
                  reference='doi:10.1103/PhysRevE.70.046204',
                  **controls, n_steps_requested=int(n_steps),steps_completed=steps,
                  converged=converged,residual=residual,trace=trace,
                  target_up_pixels=count,magnetization_constrained=target_magnetization is not None, up_fraction_measured=float(np.mean(pattern>0)),
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
    max_length_iterations=5, progress_callback=None, **controls,
):
    """Generate +/-1 domains with optional magnetization and spacing constraints.

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
    A numeric magnetization target constrains sign area at every step; None
    disables that constraint and records the achieved mean and occupancy.
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
    accumulated_timings = {}
    for attempt in range(int(max_length_iterations)):
        if progress_callback is not None:
            progress_callback(dict(stage='attempt', attempt=attempt+1,
                max_attempts=int(max_length_iterations), preferred_length_px=preferred))
        pattern,meta=_relax_constrained_domains(sz_array,preferred,
            progress_callback=progress_callback,**controls)
        for name, seconds in meta['timings_seconds'].items():
            accumulated_timings[name] = accumulated_timings.get(name, 0.) + seconds
        measured=meta['measurements']['radial_period_px']
        error=abs(measured/requested-1) if measured is not None else float('inf')
        history.append(dict(attempt=attempt+1, preferred_length_px=preferred,
                            measured_length_px=measured, relative_error=error if measured is not None else None,
                            steps_completed=meta['steps_completed'],converged=meta['converged'],
                            timings_seconds=dict(meta['timings_seconds'])))
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
                length_tolerance=float(length_tolerance),max_length_iterations=int(max_length_iterations),
                calibration_enabled=bool(max_length_iterations > 1),calibration_history=history,
                elapsed_seconds=perf_counter()-start)
    accumulated_timings['total'] = perf_counter() - start
    meta['timings_seconds'] = accumulated_timings
    if progress_callback is not None:
        progress_callback(dict(stage='complete', elapsed_seconds=accumulated_timings['total'],
            timings_seconds=dict(accumulated_timings)))
    return pattern,meta

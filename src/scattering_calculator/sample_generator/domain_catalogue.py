"""Measured scalar morphology descriptors for reproducible texture catalogues."""
import numpy as np
from .domain_analysis import classify_magnetic_domains


def measure_domain_pattern(pattern, *, classification=None, **classification_options):
    """Measure occupancy, component shapes, and local sixfold center order.

    An optional precomputed ``classification`` must describe the same pattern
    and selection; it avoids repeating component analysis. Additional classifier
    options cannot be supplied with a precomputed result.
    Shape and boundary options are forwarded to ``classify_magnetic_domains``.
    Only classified bubbles contribute to center-order measurements. Sixfold
    order is a bubble-center bond statistic, not proof of a crystal or magnetic
    topology. Mean shape descriptors retain their minority-component convention;
    counts and orientational order include both polarities. Component classification uses the existing pixel-based
    classifier and therefore depends on resolution and its default thresholds.
    """
    values = np.asarray(pattern, dtype=float)
    centered = values - values.mean()
    power = np.abs(np.fft.fft2(centered))**2
    power[0, 0] = 0
    peak_y, peak_x = np.unravel_index(np.argmax(power), power.shape)
    frequency = np.hypot(np.fft.fftfreq(values.shape[0])[peak_y],
                         np.fft.fftfreq(values.shape[1])[peak_x])
    period = float(1 / frequency) if power.max() > 0 and frequency > 0 else None
    if classification is not None and classification_options:
        raise ValueError("Pass classifier options or a precomputed classification, not both")
    analysis = classification if classification is not None else classify_magnetic_domains(values, **classification_options)
    components = [c for c in analysis['minority_components'] if c['area'] >= analysis['classification_min_area_px'] and not c['is_boundary_fragment']]
    order = analysis['sixfold_order']
    return dict(dominant_period_px=period, up_fraction=float(np.mean(values > 0)), mean_mz=float(values.mean()),
                state=analysis['state'], stripe_nematic_order=analysis['stripe_nematic_order'],
                local_sixfold_order=analysis['local_sixfold_order'],
                morphology=analysis['morphology'], bubble_count=analysis['bubble_count'],
                stripe_count=analysis['stripe_count'], noise_count=analysis['noise_count'],
                boundary_fragment_count=analysis['boundary_fragment_count'],
                mean_eccentricity=float(np.mean([c['eccentricity'] for c in components])) if components else None,
                mean_circularity=float(np.mean([c['circularity'] for c in components])) if components else None,
                sixfold_order=order, interior_component_count=analysis['interior_bubble_count'],
                **{f'{phase}_{kind}_count': analysis[f'{phase}_{kind}_count']
                   for phase in ('positive', 'negative', 'majority', 'minority')
                   for kind in ('bubble', 'stripe', 'noise', 'boundary_fragment')})

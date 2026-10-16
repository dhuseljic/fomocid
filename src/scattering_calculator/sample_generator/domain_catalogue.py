"""Measured scalar morphology descriptors for reproducible texture catalogues."""
import numpy as np
from scipy.spatial import KDTree
from .domain_analysis import classify_magnetic_domains


def measure_domain_pattern(pattern):
    """Measure occupancy, component shapes, and local sixfold center order.

    Border components contribute to fractions but are excluded from center-order
    measurements. Sixfold order is a local bond statistic, not proof of a crystal
    or magnetic topology. Component classification uses the existing pixel-based
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
    analysis = classify_magnetic_domains(values)
    components = [c for c in analysis['minority_components'] if c['area'] >= 9]
    centers = []
    labels = analysis['minority_label_image']
    for component in components:
        if not component['touches_border']:
            centers.append(np.argwhere(labels == component['label']).mean(axis=0))
    order = None
    if len(centers) >= 7:
        points = np.asarray(centers)
        _, neighbors = KDTree(points).query(points, k=7)
        vectors = points[neighbors[:, 1:]] - points[:, None, :]
        angles = np.arctan2(vectors[..., 0], vectors[..., 1])
        order = float(np.abs(np.mean(np.exp(6j * angles))))
    return dict(dominant_period_px=period, up_fraction=float(np.mean(values > 0)), mean_mz=float(values.mean()),
                morphology=analysis['morphology'], bubble_count=analysis['bubble_count'],
                stripe_count=analysis['stripe_count'], noise_count=analysis['noise_count'],
                mean_eccentricity=float(np.mean([c['eccentricity'] for c in components])) if components else None,
                mean_circularity=float(np.mean([c['circularity'] for c in components])) if components else None,
                sixfold_order=order, interior_component_count=len(centers))

"""Physical-plane opaque-stop diffraction using an exact-distance RS correction."""
from types import SimpleNamespace
import numpy as np
from .detector_propagation import RayleighSommerfeldPropagator
from scattering_calculator.experimental_conditions.detector import beamstop


def stop_plane_mask(config, wavelength):
    """Construct physical opacity without detector-plane magnification.

    The second-hop quadrature must resolve the carrier phase. Refuse a coarse
    grid rather than returning plausible-looking aliased fringes.
    """
    distance = config.sample_to_detector_distance
    gap = config.beamstop_config.bs_detector_distance
    if not 0 < gap < distance:
        raise ValueError('Beamstop diffraction requires 0 < beamstop-to-camera distance < sample-to-camera distance')
    z = distance-gap
    shape = tuple(config.beamstop_diffraction_shape)
    if len(shape) != 2 or any(int(n) != n or n < 8 or n > 2048 for n in shape):
        raise ValueError('beamstop_diffraction_shape needs two integers from 8 to 2048')
    detector_radius = float(np.max(np.hypot(config.detector_layout.detx, config.detector_layout.dety)))
    pitch = config.beamstop_diffraction_pixel_size
    if pitch is None:
        pitch = wavelength/(8*max(detector_radius/gap, 1e-6))
    if not np.isfinite(pitch) or pitch <= 0:
        raise ValueError('beamstop_diffraction_pixel_size must be positive')
    yy, xx = np.indices(shape, dtype=float)
    x, y = (xx-shape[1]/2)*pitch, (yy-shape[0]/2)*pitch
    radius = float(np.max(np.hypot(x, y)))
    # Conservative phase-step criterion, including incident spherical curvature.
    angular_budget = radius/np.hypot(z, radius) + (radius+detector_radius)/np.hypot(gap, radius+detector_radius)
    if pitch*angular_budget > wavelength/4:
        raise ValueError('Beamstop-plane grid undersamples phase: reduce beamstop_diffraction_pixel_size and increase grid shape to retain stop coverage')
    center = config.beamstop_config.bs_center
    detector_center = config.detector_center
    center = tuple(shape[i]/2 + (center[i]-detector_center[i])*config.pixel_size*z/distance/pitch for i in range(2))
    layout = SimpleNamespace(detector_shape=shape, pixel_size=pitch, distance_sample_detector=z)
    stop = beamstop(layout, distance_detector_beamstop=0.)
    kwargs = dict(config.beamstop_config.bs_config)
    kwargs.pop('center', None)
    if config.beamstop_config.bs_method is None:
        stop.create_empty_beamstop()
    else:
        stop.create_circle_beamstop(center=center, use_real_space_coordinates=True, **kwargs)
    opacity = np.clip(stop.beamstop, 0., 1.)
    edge = np.concatenate((opacity[0], opacity[-1], opacity[:, 0], opacity[:, -1]))
    if edge.max() > 1e-6:
        raise ValueError('Beamstop diffraction grid clips the stop or wire; enlarge beamstop_diffraction_shape at the same pitch')
    return opacity, pitch, x, y, z, gap


def propagate_with_stop(field, source_pitch, wavelength, distance, detector_x, detector_y,
                        opacity, stop_pitch, stop_x, stop_y, source_to_stop, stop_to_detector):
    """Subtract the propagated blocked field (Babinet) without truncating the open beam.

    Opacity is an intensity blocking fraction, so amplitude transmission is
    sqrt(1-opacity). RS uses the full distance and obliquity, not a Fresnel
    expansion. Grid/source-window convergence remains necessary.
    """
    direct = RayleighSommerfeldPropagator(field.shape[:2], source_pitch, wavelength,
                                        distance, detector_x, detector_y).forward(field)
    if not np.any(opacity):
        return direct
    at_stop = RayleighSommerfeldPropagator(field.shape[:2], source_pitch, wavelength,
                                         source_to_stop, stop_x, stop_y).forward(field)
    blocked_fraction = 1-np.sqrt(1-opacity)
    blocked = at_stop*blocked_fraction.reshape(opacity.shape+(1,)*(field.ndim-2))
    correction = RayleighSommerfeldPropagator(opacity.shape, stop_pitch, wavelength,
        stop_to_detector, detector_x, detector_y).forward(blocked)
    return direct-correction

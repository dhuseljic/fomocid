"""Reproducible paper experiments. All lengths here are nm unless marked otherwise.

The Born reference is deliberately separate from the production propagators.
Run from any directory: python /path/to/experiments.py --experiment all
"""
from pathlib import Path
from types import SimpleNamespace
from dataclasses import dataclass, asdict, field, replace
import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time

import numpy as np
import scipy
from scipy.interpolate import RegularGridInterpolator
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from scattering_calculator.database.database_loading import material_params
from scattering_calculator.sample_generator.structures import Structure
from scattering_calculator.beam_propagator.simple_propagation import scalar_wavefronts
from scattering_calculator.beam_propagator.Jones_propagator import wavefronts
from scattering_calculator.beam_propagator.Stokes_propagator import stokes_wavefronts

OUT = Path(__file__).resolve().parent / 'results'
HC = 1239.8419843320026  # eV nm


@dataclass(frozen=True)
class SkyrmionConfig:
    """Notebook-editable geometry; lengths nm except explicitly marked metres.

    Radius is the compact texture radius (core to fully +z background).
    Tilt is about lab y, measured from normal incidence. None thickness selects
    the second-zero design; an explicit thickness is never silently retuned.
    """
    energy_eV: float = 778.
    lattice_nm: float = 12.
    radius_nm: float = 3.4
    thickness_nm: float | None = None
    angles_deg: tuple = (-8.7947589, -4.3973795, 0., 4.3973795, 8.7947589)
    scan_angles_deg: tuple = field(default_factory=lambda: tuple(np.linspace(-12, 12, 49)))
    detector_n: int = 193
    detector_pitch_m: float = 13.5e-6
    detector_distance_m: float = .007
    detector_center_offset_xy_px: tuple = (0., 0.)
    beam_sigma_nm: float = 24.
    born_n: int = 256
    born_dx_nm: float = .75
    multislice_n: int = 192
    multislice_dx_nm: float = .75
    multislice_dz_nm: float = 2.
    volume_n: int = 128
    volume_angles_deg: tuple = field(default_factory=lambda: tuple(np.linspace(-12, 12, 25)))
    q_bins: int = 101
    q_limit_rad_nm: float = .85
    fft_nz: int = 512
    fft_dz_nm: float = 2.
    contrast_channel: str = 'xmcd'  # 'mz' gives an orientation-independent scalar control
    include_cobalt: bool = True

    def __post_init__(self):
        positive = ('energy_eV', 'lattice_nm', 'radius_nm', 'detector_pitch_m',
                    'detector_distance_m', 'beam_sigma_nm', 'born_dx_nm',
                    'multislice_dx_nm', 'multislice_dz_nm', 'q_limit_rad_nm', 'fft_dz_nm')
        for name in positive:
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be finite and positive')
        for name in ('detector_n', 'born_n', 'multislice_n', 'volume_n', 'q_bins', 'fft_nz'):
            value = getattr(self, name)
            if not isinstance(value, (int, np.integer)) or value < 3:
                raise ValueError(f'{name} must be an integer >= 3')
        if self.radius_nm >= self.lattice_nm/2:
            raise ValueError('radius_nm must be < lattice_nm/2 for non-overlapping tubes')
        if self.thickness_nm is not None and (not np.isfinite(self.thickness_nm) or self.thickness_nm <= 0):
            raise ValueError('thickness_nm must be positive or None')
        if self.contrast_channel not in ('xmcd', 'mz'):
            raise ValueError("contrast_channel must be 'xmcd' or 'mz'")
        for name in ('angles_deg', 'scan_angles_deg', 'volume_angles_deg'):
            angles = np.asarray(getattr(self, name), float)
            if angles.ndim != 1 or not angles.size or not np.all(np.isfinite(angles)) or np.any(abs(angles) >= 80):
                raise ValueError(f'{name} must contain finite tilts strictly between -80 and 80 degrees')
        if len(self.detector_center_offset_xy_px) != 2 or not np.all(np.isfinite(self.detector_center_offset_xy_px)):
            raise ValueError('detector_center_offset_xy_px must contain two finite offsets')
        design(self.energy_eV, self.lattice_nm, self.thickness_nm)

    def geometry(self):
        return design(self.energy_eV, self.lattice_nm, self.thickness_nm)


@dataclass(frozen=True)
class FTHConfig:
    """FTH specimen, illumination and numerical grid; dimensions in nm."""
    energy_eV: float = 778.
    n: int = 192
    dx_nm: float = 10.
    layers_nm: tuple = (('Au', 300.), ('SiN', 20.), ('Pt', 2.), ('Co', 10.), ('Pt', 2.))
    max_slice_nm: float = 5.
    object_radius_nm: float = 180.
    reference_radius_nm: float = 30.
    reference_xy_nm: tuple = (600., 0.)
    beam_sigma_nm: float = 650.
    domain_period_nm: float = 110.
    propagate: bool = True
    energies_eV: tuple = (772., 775., 778., 780., 783., 790., 795., 800.)

    def __post_init__(self):
        for name in ('energy_eV', 'dx_nm', 'max_slice_nm', 'object_radius_nm',
                     'reference_radius_nm', 'beam_sigma_nm', 'domain_period_nm'):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be positive')
        if not isinstance(self.n, (int, np.integer)) or self.n < 8:
            raise ValueError('n must be an integer >= 8')
        if len(self.reference_xy_nm) != 2 or not np.all(np.isfinite(self.reference_xy_nm)):
            raise ValueError('reference_xy_nm must contain two finite coordinates')
        half = (self.n/2-1)*self.dx_nm
        if self.object_radius_nm >= half or max(abs(np.asarray(self.reference_xy_nm)))+self.reference_radius_nm >= half:
            raise ValueError('Both apertures must fit inside the real-space grid')
        if np.linalg.norm(self.reference_xy_nm) <= 3*self.object_radius_nm+self.reference_radius_nm:
            raise ValueError('Reference is too close to separate the FTH sideband from autocorrelation')
        if not self.layers_nm or any(t <= 0 for _, t in self.layers_nm):
            raise ValueError('layers_nm must contain positive material thicknesses')


@dataclass(frozen=True)
class DetectorEffectsConfig:
    """Synthetic expected count scale and production detector model settings."""
    expected_peak_counts: float = 2e4
    beamstop_radius_px: float = 3.
    noise_seed: int = 17
    measurement: dict = field(default_factory=lambda: dict(number_frames=1, exposure_time=1., max_counts_per_image=None))
    detector: dict = field(default_factory=lambda: dict(counts_per_photon=1., quantum_efficiency=1.,
        readout_noise_average=0., readout_noise_sigma=2., detector_threshold=16000.))
    artifacts: dict = field(default_factory=lambda: dict(sigma_photon=0., camera_seed=31,
        average_hot_pixels=12., average_cold_pixels=12., cosmic_rays_per_second=2.))


def savefig(fig, name, out=OUT):
    out.mkdir(parents=True, exist_ok=True)
    for suffix in ('png', 'pdf'):
        fig.savefig(out / f'{name}.{suffix}', dpi=180, bbox_inches='tight')
    plt.close(fig)


def provenance(out, parameters):
    out.mkdir(parents=True, exist_ok=True)
    db = ROOT / 'src/scattering_calculator/database/material_parameter/refractive_indexes'
    files = [*db.glob('Co/*.txt'), *db.glob('Pt/*.txt'), *db.glob('Au/*.txt'), *db.glob('Si3N4/*.txt')]
    record = dict(parameters=parameters, python=platform.python_version(), numpy=np.__version__,
                  scipy=scipy.__version__, matplotlib=matplotlib.__version__,
                  git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                  git_status=subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True),
                  database_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                  source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in [Path(__file__), * (ROOT / 'src/scattering_calculator').rglob('*.py')]})
    (out / 'provenance.json').write_text(json.dumps(record, indent=2))


def rotation_y(theta):
    """Active sample-to-laboratory rotation; theta in degrees, xyz vectors."""
    c, s = np.cos(np.deg2rad(theta)), np.sin(np.deg2rad(theta))
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def ewald_pixels(n=193, energy=778., pitch=13.5e-6, distance=.007, center_offset_xy=(0., 0.)):
    """Flat detector -> q_lab in rad/nm and pixel solid angle in sr."""
    u = (np.arange(n) - (n - 1) / 2) * pitch
    x, y = np.meshgrid(u-center_offset_xy[0]*pitch, u-center_offset_xy[1]*pitch)
    r = np.sqrt(x*x + y*y + distance**2)
    k = 2*np.pi*energy/HC
    q = k * np.stack((x/r, y/r, distance/r - 1), axis=-1)
    domega = pitch**2 * distance / r**3
    return q, domega


def design(energy=778., a=12., thickness=None):
    k = 2*np.pi*energy/HC
    g = 4*np.pi/(np.sqrt(3)*a)
    if energy <= 0 or a <= 0 or g >= k:
        raise ValueError('Require positive energy/lattice spacing and first-order G < k')
    mismatch = g*g/(k+np.sqrt(k*k-g*g))
    t = 4*np.pi/mismatch if thickness is None else float(thickness)
    if not np.isfinite(t) or t <= 0:
        raise ValueError('thickness must be finite and positive')
    bragg = np.rad2deg(np.arcsin(g/(2*k)))
    return dict(energy_eV=energy, wavelength_nm=HC/energy, a_nm=a, G_rad_nm=g,
                thickness_nm=t, mismatch_rad_nm=mismatch, first_zero_rad_nm=2*np.pi/t,
                bragg_deg=bragg, normal_thickness_factor=float(np.sinc(mismatch*t/(2*np.pi))**2),
                outside_central_lobe=bool(mismatch > 2*np.pi/t))


def texture(x, y, a=12., radius=3.4):
    """Unit Neel tubes on a triangular lattice; compact smooth walls, core -z.

    a1=(sqrt(3)*a/2,a/2), a2=(0,a). Nearest of 3x3 cell candidates.
    This is a prescribed texture, not a micromagnetic equilibrium calculation.
    """
    x, y = np.broadcast_arrays(x, y)
    i0 = np.rint(x/(np.sqrt(3)*a/2)).astype(int)
    j0 = np.rint(y/a-i0/2).astype(int)
    best = np.full(x.shape, np.inf)
    ux, uy = np.zeros(x.shape), np.zeros(x.shape)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            i, j = i0+di, j0+dj
            xx, yy = x-i*np.sqrt(3)*a/2, y-(j+i/2)*a
            r2 = xx*xx+yy*yy
            use = r2 < best
            ux, uy, best = np.where(use, xx, ux), np.where(use, yy, uy), np.minimum(best, r2)
    r = np.sqrt(best)
    # theta=pi at core, 0 beyond radius; zero derivative at both endpoints.
    u = np.clip(r/radius, 0, 1)
    angle = np.pi*(1-3*u*u+2*u*u*u)
    phi = np.arctan2(uy, ux)
    return np.stack((np.sin(angle)*np.cos(phi), np.sin(angle)*np.sin(phi), np.cos(angle)), axis=-1)


def born_interpolators(n=512, dx=.5, a=12., sigma=32., radius=3.4):
    u = (np.arange(n)-n//2)*dx
    x, y = np.meshgrid(u, u)
    m = texture(x, y, a, radius)
    m[..., 2] -= 1  # magnetic contrast against uniformly saturated background
    envelope = np.exp(-(x*x+y*y)/(2*sigma*sigma))
    f = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(m*envelope[..., None], axes=(0, 1)), axes=(0, 1)), axes=(0, 1))*dx**2
    q = np.fft.fftshift(np.fft.fftfreq(n, dx))*2*np.pi
    return [RegularGridInterpolator((q, q), f[..., c], bounds_error=False, fill_value=0) for c in range(3)]


def born_pattern(q_lab, theta, t, transforms, channel='xmcd'):
    """First-Born scalar XMCD channel m_lab dot k_in; arbitrary differential units."""
    r = rotation_y(theta)
    qs = q_lab @ r  # row-vector equivalent of R.T @ q_lab
    points = np.stack((qs[..., 1], qs[..., 0]), axis=-1)
    k_sample = r.T @ np.array([0., 0., 1.]) if channel == 'xmcd' else np.array([0., 0., 1.])
    amplitude = sum(k_sample[c]*transforms[c](points) for c in range(3))
    amplitude *= t*np.sinc(qs[..., 2]*t/(2*np.pi))
    return np.abs(amplitude)**2, qs


def grid_intensities(qs, intensity, bins=101, limit=.85):
    """Coverage-normalized bin average; unmeasured voxels are NaN, not zero."""
    edges = [np.linspace(-limit, limit, bins+1)]*3
    points = np.concatenate([q.reshape(-1, 3) for q in qs])
    values = np.concatenate([np.asarray(i).ravel() for i in intensity])
    valid = np.isfinite(values) & np.all(np.isfinite(points), axis=1)
    points, values = points[valid], values[valid]
    sums, _ = np.histogramdd(points, bins=edges, weights=values)
    hits, _ = np.histogramdd(points, bins=edges)
    mean = np.divide(sums, hits, out=np.full_like(sums, np.nan), where=hits > 0)
    return mean, hits, edges[0]


def skyrmion_born(out=OUT, config=None):
    out.mkdir(parents=True, exist_ok=True)
    cfg = config or SkyrmionConfig()
    p = cfg.geometry()
    provenance(out, asdict(cfg))
    q, domega = configured_detector(cfg)
    transforms = configured_transforms(cfg)
    theta_b = p['bragg_deg']
    angles = cfg.angles_deg
    patterns = [born_pattern(q, a, p['thickness_nm'], transforms, cfg.contrast_channel)[0] for a in angles]
    vmax = np.max(patterns)
    fig, axes = plt.subplots(1, len(angles), figsize=(3.2*len(angles), 3.5), constrained_layout=True, squeeze=False)
    axes = axes.ravel()
    for ax, a, im in zip(axes, angles, patterns):
        image = ax.imshow(im/vmax, origin='lower', norm=LogNorm(1e-6, 1), extent=(-cfg.detector_n/2,cfg.detector_n/2,-cfg.detector_n/2,cfg.detector_n/2))
        ax.set(title=f'{a:+.2f}°', xlabel='detector column offset', ylabel='row offset')
    fig.colorbar(image, ax=axes, label='differential intensity / shared maximum', shrink=.75)
    savefig(fig, 'fig02_skyrmion_tilts', out)

    dense_angles = cfg.scan_angles_deg
    intensities, coordinates = [], []
    for theta in dense_angles:
        im, qs = born_pattern(q, theta, p['thickness_nm'], transforms, cfg.contrast_channel)
        # Simulated pixel signal C=I*dOmega; undo acceptance before gridding.
        counts = im*domega
        intensities.append(counts/domega)
        coordinates.append(qs)
    volume, coverage, edges = grid_intensities(coordinates, intensities, cfg.q_bins, cfg.q_limit_rad_nm)
    np.savez_compressed(out/'reciprocal_volume.npz', intensity=volume, coverage=coverage,
                        q_edges_rad_nm=edges, angles_deg=dense_angles, config_json=json.dumps(asdict(cfg)))
    np.savez_compressed(out/'tilt_patterns.npz', intensity=patterns, angles_deg=angles,
                        q_lab_rad_nm=q, solid_angle_sr=domega)
    centres = (edges[1:]+edges[:-1])/2
    # Intensity maximum over q_y, with a separate coverage projection.
    projected = np.max(np.where(coverage>0, volume, -np.inf), axis=1)
    projected[~np.isfinite(projected)] = np.nan
    fig = plt.figure(figsize=(12, 4), constrained_layout=True)
    ax = fig.add_subplot(131, projection='3d')
    valid = (coverage>0) & (volume > np.nanmax(volume)*.003)
    ix, iy, iz = np.where(valid)
    ax.scatter(centres[ix], centres[iy], centres[iz], c=np.log10(volume[valid]/np.nanmax(volume)), s=2)
    ax.set(xlabel='$q_x$ (rad/nm)', ylabel='$q_y$', zlabel='$q_z$', title='Measured Ewald samples')
    ax = fig.add_subplot(132)
    ax.imshow(projected.T/np.nanmax(volume), origin='lower', extent=(-cfg.q_limit_rad_nm,cfg.q_limit_rad_nm,-cfg.q_limit_rad_nm,cfg.q_limit_rad_nm), norm=LogNorm(1e-6,1), aspect='auto')
    ax.set(xlabel='$q_x$ (rad/nm)', ylabel='$q_z$ (rad/nm)', title='Maximum over $q_y$')
    ax = fig.add_subplot(133)
    ax.imshow(coverage.sum(axis=1).T, origin='lower', extent=(-cfg.q_limit_rad_nm,cfg.q_limit_rad_nm,-cfg.q_limit_rad_nm,cfg.q_limit_rad_nm), aspect='auto')
    ax.set(xlabel='$q_x$ (rad/nm)', ylabel='$q_z$ (rad/nm)', title='Sampling coverage (sum over $q_y$)')
    savefig(fig, 'fig03_reciprocal_space', out)

    # Exact centre of +G rod, solving elastic scattering for longitudinal q_z.
    k, g, t = 2*np.pi/p['wavelength_nm'], p['G_rad_nm'], p['thickness_nm']
    a = np.deg2rad(dense_angles)
    qz = -k*np.cos(a)+np.sqrt((k*np.cos(a))**2-g*g+2*k*g*np.sin(a))
    rocking = np.sinc(qz*t/(2*np.pi))**2
    thin = np.sinc(qz*20/(2*np.pi))**2
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
    u = (np.arange(192)-96)*.5
    x,y = np.meshgrid(u,u)
    axes[0].imshow(texture(x,y,cfg.lattice_nm,cfg.radius_nm)[...,2], origin='lower', extent=(-48,48,-48,48), cmap='RdBu_r', vmin=-1, vmax=1)
    axes[0].set(xlabel='x (nm)', ylabel='y (nm)', title='Prescribed Neel skyrmion tubes')
    axes[1].plot(dense_angles, rocking, label=f'{t:.1f} nm'); axes[1].plot(dense_angles, thin, label='20 nm')
    axes[1].axvline(theta_b, color='gray', ls=':'); axes[1].legend()
    axes[1].set(xlabel='sample tilt (degrees)', ylabel='longitudinal intensity factor', title='+G rod rocking curve')
    qz_axis = np.linspace(-.15,.15,1000)
    axes[2].plot(qz_axis, np.sinc(qz_axis*t/(2*np.pi))**2)
    axes[2].axvline(-p['mismatch_rad_nm'], color='red', label='untilted Ewald intersection')
    axes[2].legend(fontsize=7); axes[2].set(xlabel='$q_z$ (rad/nm)', ylabel='thickness envelope')
    savefig(fig, 'fig01_geometry', out)
    (out/'geometry.json').write_text(json.dumps(p, indent=2))
    print(json.dumps(p, indent=2))
    return p


def skyrmion_multislice(theta, n=None, dx=None, dz=None, material='weak', propagate=True, thickness=None, config=None):
    """Production scalar propagation through a rotated volume, in SI units.

    Weak medium isolates geometry. Co uses the bundled tabulation without scaling.
    Return physical q coordinates: with the code's exp(-ikz z) convention,
    transverse physical outgoing momenta are the negative FFT frequencies.
    """
    cfg = config or SkyrmionConfig()
    n = cfg.multislice_n if n is None else n
    dx = cfg.multislice_dx_nm if dx is None else dx
    dz = cfg.multislice_dz_nm if dz is None else dz
    p = cfg.geometry(); t = p['thickness_nm'] if thickness is None else thickness
    r = rotation_y(theta)
    u = (np.arange(n)-n//2)*dx
    x,y = np.meshgrid(u,u)
    zextent = t/np.cos(np.deg2rad(theta)) + n*dx*abs(np.tan(np.deg2rad(theta)))+4*dz
    nz = int(np.ceil(zextent/dz)); z = (np.arange(nz)-(nz-1)/2)*dz
    xx = r[0,0]*x[None] + r[2,0]*z[:,None,None]
    depth = r[0,2]*x[None] + r[2,2]*z[:,None,None]
    m = texture(xx, y[None], cfg.lattice_nm, cfg.radius_nm)
    mz_lab = r[2,0]*m[...,0]+r[2,2]*m[...,2] if cfg.contrast_channel == 'xmcd' else m[...,2]
    saturation = r[2,2] if cfg.contrast_channel == 'xmcd' else 1.
    # Fractional occupancy along z at interfaces (exact for a planar slab).
    width = dz*abs(r[2,2])
    fraction = np.clip((t/2 - depth + width/2)/width,0,1)-np.clip((-t/2-depth+width/2)/width,0,1)
    if material == 'Co':
        n0,nc,_ = material_params.load_refractive_index('Co',p['energy_eV'])
    elif material == 'weak':
        n0,nc = 1+0j, 1e-6+0j
    else:
        raise ValueError('material must be weak or Co')
    stack = 1+fraction*((n0-1)+nc*mz_lab)
    beam = np.exp(-(x*x+y*y)/(2*cfg.beam_sigma_nm**2)).astype(complex)
    options = dict(beam_parameters=SimpleNamespace(wavelength=p['wavelength_nm']*1e-9,pol='CR'),
                   layer_thicknesses=np.full(nz,dz*1e-9),real_space_pixel_size=dx*1e-9,
                   E_in=beam,propagate=propagate)
    field = scalar_wavefronts(refractive_index_stack=stack,**options).exit_wave
    # Subtract a coherent saturated reference to expose the magnetic channel.
    # This is a computed complex-field difference, NOT a measured helicity difference.
    reference = 1+fraction*((n0-1)+nc*saturation)
    bg = scalar_wavefronts(refractive_index_stack=reference,**options).exit_wave
    f = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field-bg)))*dx**2
    q = -2*np.pi*np.fft.fftshift(np.fft.fftfreq(n,dx))
    qx,qy = np.meshgrid(q,q)
    k = 2*np.pi/p['wavelength_nm']
    valid = qx*qx+qy*qy < k*k
    qz = np.sqrt(np.maximum(k*k-qx*qx-qy*qy,0))-k
    qlab = np.stack((qx,qy,qz),axis=-1)
    im = np.abs(f)**2
    im[~valid] = np.nan
    return dict(intensity=im,q_lab=qlab,transmission=float(np.sum(abs(field)**2)/np.sum(abs(beam)**2)),
                n=n,dx_nm=dx,dz_nm=dz,theta_deg=theta,material=material, amplitude=f)


def configured_detector(cfg):
    """Physical flat-detector sampling shared by Born, multislice and FFT checks."""
    return ewald_pixels(cfg.detector_n, cfg.energy_eV, cfg.detector_pitch_m,
                        cfg.detector_distance_m, cfg.detector_center_offset_xy_px)


def configured_transforms(cfg):
    return born_interpolators(cfg.born_n, cfg.born_dx_nm, cfg.lattice_nm,
                              cfg.beam_sigma_nm, cfg.radius_nm)


def sample_multislice_detector(result, cfg):
    """Interpolate the complex exit spectrum onto the configured Ewald pixels.

    Outside the source FFT bandwidth is NaN (unmeasured), never extrapolated.
    Values are angular-spectrum diagnostics, not absolute detector counts.
    """
    q, _ = configured_detector(cfg)
    axis = result['q_lab'][0, :, 0][::-1]
    interp = RegularGridInterpolator((axis, axis), result['amplitude'][::-1, ::-1],
                                     bounds_error=False, fill_value=np.nan)
    amplitude = interp(np.stack((q[..., 1], q[..., 0]), axis=-1))
    return dict(result, intensity=abs(amplitude)**2, q_lab=q, amplitude=amplitude)


def multislice_figure(out=OUT, config=None):
    cfg = config or SkyrmionConfig()
    out.mkdir(parents=True, exist_ok=True)
    provenance(out, asdict(cfg))
    angles = cfg.angles_deg
    materials = ['weak', 'Co'] if cfg.include_cobalt else ['weak']
    all_cases = {}
    for material in materials:
        cases = [sample_multislice_detector(skyrmion_multislice(a, material=material, config=cfg), cfg)
                 for a in angles]
        all_cases[material] = cases
        vmax = np.nanmax([c['intensity'] for c in cases])
        fig, axes = plt.subplots(1, len(angles), figsize=(3.6*len(angles), 3.5),
                                constrained_layout=True, squeeze=False)
        axes = axes.ravel()
        for ax, c in zip(axes, cases):
            q = c['q_lab']
            image = ax.pcolormesh(q[...,0], q[...,1], c['intensity']/vmax,
                                 norm=LogNorm(1e-6,1), shading='auto')
            ax.set(xlabel='$q_x$ (rad/nm)', ylabel='$q_y$ (rad/nm)',
                   title=f"{material} {c['theta_deg']:.2f}°; T={c['transmission']:.3f}")
        fig.colorbar(image, ax=axes, label=f'intensity / shared {material} maximum', shrink=.75)
        name = 'fig04_multislice_tilts' if material == 'weak' else 'fig04b_cobalt_tilts'
        savefig(fig, name, out)
        name = 'multislice_tilts' if material == 'weak' else 'cobalt_tilts'
        np.savez_compressed(out/f'{name}.npz', intensity=[c['intensity'] for c in cases],
                            q_lab=cases[0]['q_lab'], angles_deg=angles,
                            transmission=[c['transmission'] for c in cases])
    coords, signals = [], []
    for angle in cfg.volume_angles_deg:
        result = sample_multislice_detector(skyrmion_multislice(angle, n=cfg.volume_n, config=cfg), cfg)
        coords.append(result['q_lab'] @ rotation_y(angle))
        signals.append(result['intensity'])
    volume, hits, edges = grid_intensities(coords, signals, cfg.q_bins, cfg.q_limit_rad_nm)
    np.savez_compressed(out/'multislice_reciprocal_volume.npz', intensity=volume, coverage=hits,
                        q_edges_rad_nm=edges, angles_deg=cfg.volume_angles_deg, config_json=json.dumps(asdict(cfg)))
    centres = (edges[:-1]+edges[1:])/2
    valid = (hits>0) & (volume>np.nanmax(volume)*.003)
    ix, iy, iz = np.where(valid)
    fig = plt.figure(figsize=(7,5)); ax = fig.add_subplot(projection='3d')
    ax.scatter(centres[ix], centres[iy], centres[iz], c=np.log10(volume[valid]/np.nanmax(volume)), s=4)
    ax.set(xlabel='$q_x$ (rad/nm)', ylabel='$q_y$ (rad/nm)', zlabel='$q_z$ (rad/nm)',
           title='Scalar multislice: configured detector and tilts')
    savefig(fig, 'fig04c_multislice_volume', out)
    return all_cases['weak']


def fft_volume_comparison(out=OUT, config=None):
    """Compare Ewald-assembled diffraction to actual FFTs of a voxelized 3D image.

    Array order is z,y,x for the image/FFTs, x,y,z for gridded intensities.
    Three separate scipy.fft.fftn calls avoid retaining a dense vector FFT.
    The image is the same Gaussian-weighted m-(0,0,1) contrast as the Born
    calculation, with fractional z-face occupancy. FFT values include dx² dz.
    """
    from scipy.fft import fftn, fftshift, ifftshift
    cfg = config or SkyrmionConfig()
    out.mkdir(parents=True, exist_ok=True)
    provenance(out, asdict(cfg))
    p = cfg.geometry()
    n, dx, nz, dz = cfg.born_n, cfg.born_dx_nm, cfg.fft_nz, cfg.fft_dz_nm
    t = p['thickness_nm']
    if nz*dz < t+4*dz:
        raise ValueError('Increase fft_nz or fft_dz_nm: the z FFT box must contain the slab plus vacuum')
    if cfg.q_limit_rad_nm >= min(np.pi/dx, np.pi/dz):
        raise ValueError('q_limit_rad_nm must lie inside both transverse and longitudinal FFT Nyquist limits')
    u = (np.arange(n)-n//2)*dx
    z = (np.arange(nz)-nz//2)*dz
    x, y = np.meshgrid(u, u)
    m = texture(x, y, cfg.lattice_nm, cfg.radius_nm)
    m[...,2] -= 1
    envelope = np.exp(-(x*x+y*y)/(2*cfg.beam_sigma_nm**2))
    occupancy = np.clip((t/2-z+dz/2)/dz,0,1)-np.clip((-t/2-z+dz/2)/dz,0,1)
    qxy = fftshift(np.fft.fftfreq(n, dx))*2*np.pi
    qz = fftshift(np.fft.fftfreq(nz, dz))*2*np.pi
    qlab, _ = configured_detector(cfg)
    angles = cfg.scan_angles_deg
    coordinates = [qlab @ rotation_y(a) for a in angles]
    amplitudes = [np.zeros(qlab.shape[:2], complex) for _ in angles]
    # A common histogram grid for the direct FFT and assembled diffraction.
    edges = np.linspace(-cfg.q_limit_rad_nm, cfg.q_limit_rad_nm, cfg.q_bins+1)
    centres = (edges[:-1]+edges[1:])/2
    gx, gy, gz = np.meshgrid(centres, centres, centres, indexing='ij')
    points = np.stack((gz, gy, gx), axis=-1)
    direct_mz = None
    ms_path = out/'multislice_reciprocal_volume.npz'
    ms = None
    if ms_path.exists():
        with np.load(ms_path) as archive:
            if 'config_json' in archive and json.loads(str(archive['config_json'])) == json.loads(json.dumps(asdict(cfg))):
                ms = {key: archive[key] for key in ('intensity', 'coverage', 'angles_deg')}
    ms_coords = [qlab @ rotation_y(a) for a in cfg.volume_angles_deg] if ms is not None else []
    ms_amplitudes = [np.zeros(qlab.shape[:2], complex) for _ in ms_coords]
    for component in range(3):
        # A genuine, explicitly voxelized 3D image, not a 2D FFT times analytic sinc.
        image3d = (occupancy[:,None,None]*envelope[None]*m[None,...,component]).astype(np.float32)
        spectrum = fftshift(fftn(ifftshift(image3d), workers=1)) * (dx*dx*dz)
        del image3d
        interp = RegularGridInterpolator((qz, qxy, qxy), spectrum, bounds_error=False, fill_value=np.nan)
        if component == 2:
            direct_mz = abs(interp(points))**2
            # Store a real-space x-z section and the full 3D reciprocal intensity
            # on the comparison grid; avoid saving gigabytes of padded FFT arrays.
            image_xz = occupancy[:,None]*envelope[n//2][None]*m[n//2,:,2][None]
        for aa, coords, amp in ((angles, coordinates, amplitudes),
                                (cfg.volume_angles_deg, ms_coords, ms_amplitudes)):
            for angle, qs, target in zip(aa, coords, amp):
                direction = rotation_y(angle).T @ [0.,0.,1.] if cfg.contrast_channel == 'xmcd' else np.array([0.,0.,1.])
                if abs(direction[component]) > 1e-15:
                    target += direction[component]*interp(qs[..., [2,1,0]])
        del interp, spectrum
    transforms = configured_transforms(cfg)
    born = [born_pattern(qlab, a, t, transforms, cfg.contrast_channel)[0] for a in angles]
    observed, hits, _ = grid_intensities(coordinates, born, cfg.q_bins, cfg.q_limit_rad_nm)
    matched, fft_hits, _ = grid_intensities(coordinates, [abs(a)**2 for a in amplitudes],
                                          cfg.q_bins, cfg.q_limit_rad_nm)
    common = (hits>0) & (fft_hits>0) & np.isfinite(matched)
    error = float(np.linalg.norm((observed-matched)[common])/np.linalg.norm(observed[common]))
    # The full direct-mz volume is a scalar control. XMCD comparisons use the
    # coherent vector projection before squaring, not a sum of component powers.
    maximum = max(np.nanmax(direct_mz), np.nanmax(observed), np.nanmax(matched))
    fig, axes = plt.subplots(2,3,figsize=(14,8),constrained_layout=True)
    extent = (-cfg.q_limit_rad_nm,cfg.q_limit_rad_nm)*2
    mid = cfg.q_bins//2
    for ax, arr, title in zip(axes[0], (direct_mz, observed, matched),
        ('Full 3D FFT: |F[m_z − 1]|²', 'Diffraction assembled on Ewald spheres', '3D FFT sampled on the same Ewald spheres')):
        plot = ax.imshow(arr[:,mid,:].T/maximum, origin='lower', extent=extent,
                         norm=LogNorm(1e-6,1), aspect='auto')
        ax.set(title=title, xlabel='$q_x$ (rad/nm)', ylabel='$q_z$ (rad/nm)', ylim=(-.25,.15))
    fig.colorbar(plot, ax=list(axes[0]), label='shared intensity scale', shrink=.8)
    residual = np.where(common, (observed-matched)/maximum, np.nan)
    bound = max(float(np.nanmax(abs(residual))), 1e-12)
    plot = axes[1,0].imshow(residual[:,mid,:].T, origin='lower', extent=extent,
                           cmap='RdBu_r', vmin=-bound, vmax=bound, aspect='auto')
    axes[1,0].set(title=f'Diffraction − matched FFT; L2={error:.3g}', xlabel='$q_x$', ylabel='$q_z$', ylim=(-.25,.15))
    fig.colorbar(plot, ax=axes[1,0], shrink=.8)
    plot = axes[1,1].imshow(hits[:,mid,:].T, origin='lower', extent=extent, aspect='auto')
    axes[1,1].set(title='Coverage at $q_y ≈ 0$', xlabel='$q_x$', ylabel='$q_z$', ylim=(-.25,.15))
    fig.colorbar(plot, ax=axes[1,1], label='samples / voxel', shrink=.8)
    gi = int(np.argmin(abs(centres-p['G_rad_nm'])))
    for arr, label in ((direct_mz,'full FFT, mz'),(observed,'diffraction'),(matched,'matched FFT')):
        axes[1,2].plot(centres,arr[gi,mid,:]/maximum,'.-',label=label)
    axes[1,2].set(xlabel='$q_z$ (rad/nm)', ylabel='shared intensity scale', xlim=(-.25,.15),
                  title=f'Rod profile at qx={centres[gi]:.3f}')
    axes[1,2].legend()
    savefig(fig, 'fig09_fft_volume_comparison', out)
    # Same 3D axes and thresholds for direct vs sampled reciprocal-space views.
    fig = plt.figure(figsize=(13,4), constrained_layout=True)
    for i, (arr,title) in enumerate(((direct_mz,'Full 3D FFT (mz)'),(observed,'Ewald diffraction'),(matched,'Matched 3D FFT'))):
        ax = fig.add_subplot(1,3,i+1,projection='3d')
        valid = np.isfinite(arr) & (arr>maximum*.003)
        ix,iy,iz = np.where(valid)
        ax.scatter(centres[ix],centres[iy],centres[iz],c=np.log10(arr[valid]/maximum),
                   vmin=-3,vmax=0,s=2)
        ax.set(title=title,xlabel='$q_x$',ylabel='$q_y$',zlabel='$q_z$',
               xlim=(-cfg.q_limit_rad_nm,cfg.q_limit_rad_nm),ylim=(-cfg.q_limit_rad_nm,cfg.q_limit_rad_nm),zlim=(-.25,.15))
    savefig(fig, 'fig09b_fft_volume_3d', out)
    report = dict(born_vs_sampled_fft_relative_L2=error, measured_voxel_fraction=float(np.mean(hits>0)),
                  fft_grid_zyx=[nz,n,n], fft_spacing_zyx_nm=[dz,dx,dx],
                  voxelized_thickness_nm=float(occupancy.sum()*dz),
                  contrast_channel=cfg.contrast_channel,
                  multislice_comparison='not run: no matching multislice archive')
    payload = dict(direct_mz_intensity=direct_mz, diffraction_intensity=observed,
                   sampled_fft_intensity=matched, coverage=hits, fft_coverage=fft_hits,
                   q_edges_rad_nm=edges, sample_mz_contrast_xz=image_xz, sample_x_nm=u, sample_z_nm=z,
                   config_json=json.dumps(asdict(cfg)))
    if ms is not None:
        # Match finite source bandwidth of the multislice detector interpolation.
        source_q = -2*np.pi*np.fft.fftshift(np.fft.fftfreq(cfg.volume_n,cfg.multislice_dx_nm))
        valid = ((qlab[...,0]>=source_q.min()) & (qlab[...,0]<=source_q.max()) &
                 (qlab[...,1]>=source_q.min()) & (qlab[...,1]<=source_q.max()))
        fft_ms, _, _ = grid_intensities(ms_coords, [np.where(valid,abs(a)**2,np.nan) for a in ms_amplitudes],
                                         cfg.q_bins,cfg.q_limit_rad_nm)
        good = (ms['coverage']>0) & np.isfinite(fft_ms)
        # Fit ONE scale, retaining all relative voxel intensities; no independent
        # peak normalization. Multislice includes illumination and propagation
        # effects beyond the Born object, so exact equality is not expected.
        scale = float(np.sum(ms['intensity'][good]*fft_ms[good])/np.sum(fft_ms[good]**2))
        diff = np.where(good,ms['intensity']-scale*fft_ms,np.nan)
        ms_error = float(np.linalg.norm(diff[good])/np.linalg.norm(ms['intensity'][good]))
        vmax = np.nanmax(ms['intensity'])
        fig, axes = plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
        for ax,arr,title in zip(axes[:2],(ms['intensity'],scale*fft_ms),('Multislice Ewald volume','Matched FFT × one fitted scale')):
            ax.imshow(arr[:,mid,:].T/vmax,origin='lower',extent=extent,norm=LogNorm(1e-6,1),aspect='auto')
            ax.set(title=title,xlabel='$q_x$',ylabel='$q_z$',ylim=(-.25,.15))
        limit=max(np.nanmax(abs(diff))/vmax,1e-12)
        axes[2].imshow(diff[:,mid,:].T/vmax,origin='lower',extent=extent,cmap='RdBu_r',vmin=-limit,vmax=limit,aspect='auto')
        axes[2].set(title=f'Scaled residual; L2={ms_error:.3g}',xlabel='$q_x$',ylabel='$q_z$',ylim=(-.25,.15))
        savefig(fig,'fig09c_multislice_fft_comparison',out)
        report.update(multislice_comparison='matched configured scan',multislice_fitted_scale=scale,
                      multislice_vs_fft_shape_relative_L2=ms_error)
        payload.update(multislice_intensity=ms['intensity'],multislice_sampled_fft_intensity=fft_ms,
                       multislice_coverage=ms['coverage'])
    np.savez_compressed(out/'fft_volume_comparison.npz',**payload)
    (out/'fft_comparison_metrics.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    return report


def set_sideband_view(ax, cfg):
    rx, ry = cfg.reference_xy_nm
    half = cfg.object_radius_nm + cfg.reference_radius_nm + cfg.dx_nm
    ax.set_xlim(rx-half, rx+half)
    ax.set_ylim(ry-half, ry+half)


def fth_case(energy=None, mode='Jones', n=None, dx_nm=None, propagate=None, config=None):
    """Au mask / SiN membrane / explicit Pt-Co-Pt; real tabulated material response."""
    cfg = config or FTHConfig()
    energy = cfg.energy_eV if energy is None else energy
    n = cfg.n if n is None else n
    dx_nm = cfg.dx_nm if dx_nm is None else dx_nm
    propagate = cfg.propagate if propagate is None else propagate
    dx=dx_nm*1e-9
    mats=material_params(materials=['Au','SiN','Pt','Co'],x_ray_energy=energy)
    sample=Structure(name='paper FTH',material_params=mats,sample_shape=[0,n,n],real_space_pixel_size=dx)
    # Resolve longitudinal propagation in the mask and magnetic film.
    layers = []
    for mat, total in cfg.layers_nm:
        count = int(np.ceil(total/cfg.max_slice_nm))
        layers.extend([(mat, total/count)]*count)
    for mat,t in layers: sample.add_layer(mat,t*1e-9)
    u=(np.arange(n)-n//2)*dx_nm
    x,y=np.meshgrid(u,u)
    oh=x*x+y*y<cfg.object_radius_nm**2
    rx, ry = cfg.reference_xy_nm
    rh=(x-rx)**2+(y-ry)**2<cfg.reference_radius_nm**2
    mz=np.tanh((np.sin(2*np.pi*x/cfg.domain_period_nm+1.4*np.sin(y/90))+ .5*np.cos(2*np.pi*y/160))/.22)
    sample.mask=np.ones((len(layers),n,n))
    sample.magnetization=np.zeros((len(layers),n,n,3))
    for i,(mat,_) in enumerate(layers):
        sample.mask[i,rh]=0  # reference hole drilled through the entire stack
        if mat=='Au': sample.mask[i,oh]=0
        if mat=='Co':
            sample.magnetization[i,...,2]=mz
            sample.magnetization[i,...,0]=np.sqrt(1-mz*mz)
    beam=np.exp(-(x*x+y*y)/(2*cfg.beam_sigma_nm**2)).astype(complex)
    if mode!='Scalar':
        sample.calculate_final_dielectric_tensor(compact=True)
    images=[]; exits=[]
    for helicity,sign in [('CR',-1),('CL',1)]:
        opts=dict(beam_parameters=SimpleNamespace(wavelength=HC/energy*1e-9,pol=helicity),
                  layer_thicknesses=np.array([t for _,t in layers])*1e-9,real_space_pixel_size=dx,propagate=propagate)
        if mode=='Scalar':
            sample.calculate_final_scalar_refractive_index(helicity)
            w=scalar_wavefronts(E_in=beam,refractive_index_stack=sample.final_scalar_refractive_index,**opts)
            field=w.exit_wave
            f=np.fft.fftshift(np.fft.fft2(field,norm='ortho'))
            intensity=abs(f)**2
        else:
            factory=wavefronts if mode=='Jones' else stokes_wavefronts
            w=factory(E_in=beam[...,None]*np.array([1,sign*1j])/np.sqrt(2),eps_stack=sample.final_dielectric_tensor,**opts)
            field=w.exit_wave if mode=='Jones' else w.exit_jones
            f=np.fft.fftshift(np.fft.fft2(field,axes=(0,1),norm='ortho'),axes=(0,1))
            intensity=np.sum(abs(f)**2,axis=-1)
        images.append(intensity); exits.append(field)
    difference=images[0]-images[1]
    reconstruction=np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(difference)))
    return dict(images=np.array(images),difference=difference,reconstruction=reconstruction,
                mz=mz,object_hole=oh,reference_hole=rh,extent_nm=[u[0],u[-1],u[0],u[-1]],
                exits=np.array(exits),energy_eV=energy,dx_nm=dx_nm)


def fth_figures(out=OUT, config=None):
    cfg = config or FTHConfig()
    provenance(out, asdict(cfg))
    out.mkdir(parents=True,exist_ok=True)
    results={mode:fth_case(mode=mode, config=cfg) for mode in ('Scalar','Jones','Stokes')}
    c=results['Jones']
    fig,axes=plt.subplots(1,4,figsize=(14,3.5),constrained_layout=True)
    axes[0].imshow(c['mz']*c['object_hole'],origin='lower',extent=c['extent_nm'],cmap='RdBu_r',vmin=-1,vmax=1)
    axes[0].contour(c['reference_hole'],levels=[.5],extent=c['extent_nm'],colors='black')
    axes[0].set(title=' / '.join(f'{m}({t:g})' for m,t in cfg.layers_nm if m != 'Au'),xlabel='x (nm)',ylabel='y (nm)')
    axes[1].imshow(c['images'][0],origin='lower',norm=LogNorm(max(c['images'].max()*1e-7,1e-15),c['images'].max()))
    axes[1].set(title='CR hologram',xlabel='reciprocal pixel')
    lim=np.max(abs(c['difference']))
    axes[2].imshow(c['difference'],origin='lower',cmap='RdBu_r',vmin=-lim,vmax=lim)
    axes[2].set(title='CR − CL hologram',xlabel='reciprocal pixel')
    axes[3].imshow(abs(c['reconstruction']),origin='lower',extent=c['extent_nm'],cmap='magma',vmax=np.max(abs(c['reconstruction']))*.4)
    axes[3].set(title='|FTH(CR − CL)|',xlabel='shift (nm)')
    set_sideband_view(axes[3], cfg)
    savefig(fig,'fig05_fth',out)
    metrics={mode:float(np.linalg.norm(r['difference']-c['difference'])/np.linalg.norm(c['difference'])) for mode,r in results.items()}
    fig,axes=plt.subplots(1,3,figsize=(11,3.5),constrained_layout=True)
    for ax,(mode,r) in zip(axes,results.items()):
        ax.imshow(r['difference'],origin='lower',cmap='RdBu_r',vmin=-lim,vmax=lim)
        ax.set(title=f'{mode}; rel. L2={metrics[mode]:.2g}',xlabel='reciprocal pixel')
    savefig(fig,'fig06_modes',out)
    np.savez_compressed(out/'fth.npz',**c)
    (out/'mode_metrics.json').write_text(json.dumps(metrics,indent=2))
    # Fixed specimen/pixel grid, hence fixed reciprocal grid across energies.
    energies=np.asarray(cfg.energies_eV)
    cube=[]; recon=[]; optical=[]
    for energy in energies:
        r=fth_case(energy=energy, config=cfg)
        cube.append(r['difference']); recon.append(r['reconstruction'])
        optical.append(material_params.load_refractive_index('Co',energy))
    cube=np.array(cube); recon=np.array(recon); optical=np.array(optical)
    u=(np.arange(cube.shape[-1])-cube.shape[-1]//2)*c['dx_nm']
    x,y=np.meshgrid(u,u); roi=(x-cfg.reference_xy_nm[0])**2+(y-cfg.reference_xy_nm[1])**2<cfg.object_radius_nm**2
    signal=np.sqrt(np.sum(abs(recon[:,roi])**2,axis=1))
    fig,axes=plt.subplots(1,3,figsize=(12,3.5),constrained_layout=True)
    axes[0].plot(energies,-optical[:,1].real,label='$delta_c$'); axes[0].plot(energies,-optical[:,1].imag,label='$beta_c$'); axes[0].legend()
    axes[0].set(xlabel='energy (eV)',ylabel='circular index contribution')
    axes[1].plot(energies,signal,'o-'); axes[1].set(xlabel='energy (eV)',ylabel='RMS amplitude in FTH sideband (a.u.)')
    axes[2].imshow(abs(recon[:,cube.shape[-1]//2,:]),aspect='auto',origin='lower',extent=(u[0],u[-1],0,len(energies)))
    axes[2].set_yticks(np.arange(len(energies))+.5,energies)
    axes[2].set(xlabel='reconstruction shift x (nm)',ylabel='energy (eV)',title='Line through FTH sidebands')
    savefig(fig,'fig07_hyperspectral',out)
    np.savez_compressed(out/'hyperspectral.npz',energies_eV=energies,difference=cube,reconstruction=recon,sideband_rms=signal)
    return metrics


def validation(out=OUT, config=None):
    """Physical checks with independent geometry and production-kernel comparisons."""
    out.mkdir(parents=True,exist_ok=True)
    cfg = config or SkyrmionConfig()
    p=cfg.geometry(); k=2*np.pi/p['wavelength_nm']; q,_=configured_detector(cfg)
    np.testing.assert_allclose(np.linalg.norm(q+[0,0,k],axis=-1),k,rtol=1e-13)
    r=rotation_y(p['bragg_deg'])
    np.testing.assert_allclose(r.T@r,np.eye(3),atol=1e-14)
    g_lab=r@np.array([p['G_rad_nm'],0.,0.])
    np.testing.assert_allclose(np.linalg.norm(g_lab+[0,0,k]),k,rtol=1e-13)
    if cfg.thickness_nm is None:
        assert p['mismatch_rad_nm']>p['first_zero_rad_nm']
        assert p['normal_thickness_factor']<1e-25
    u=np.linspace(-20,20,45); x,y=np.meshgrid(u,u)
    np.testing.assert_allclose(np.linalg.norm(texture(x,y,cfg.lattice_nm,cfg.radius_nm),axis=-1),1,atol=1e-14)
    point=np.array([[[.1,.2,.3]]]); v,h,_=grid_intensities([point,point],[np.array([[2.]]),np.array([[4.]])],bins=11)
    assert np.nanmax(v)==3 and h.max()==2 and np.isnan(v[h==0]).all()
    # Longitudinal refinement at a strong rod, plus the projection approximation.
    a=p['bragg_deg']
    coarse=skyrmion_multislice(a,n=cfg.volume_n,dx=cfg.multislice_dx_nm,dz=cfg.multislice_dz_nm,config=cfg)
    fine=skyrmion_multislice(a,n=cfg.volume_n,dx=cfg.multislice_dx_nm,dz=cfg.multislice_dz_nm/2,config=cfg)
    projection=skyrmion_multislice(0,n=cfg.volume_n,dx=cfg.multislice_dx_nm,dz=cfg.multislice_dz_nm,config=cfg,propagate=False)
    normal=skyrmion_multislice(0,n=cfg.volume_n,dx=cfg.multislice_dx_nm,dz=cfg.multislice_dz_nm,config=cfg)
    g=p['G_rad_nm']; q=coarse['q_lab']
    region=(q[...,0]-g*np.cos(np.deg2rad(a)))**2+q[...,1]**2<.08**2
    normal_region=(normal['q_lab'][...,0]-g)**2+normal['q_lab'][...,1]**2<.08**2
    error=float(np.linalg.norm((coarse['intensity']-fine['intensity'])[region])/np.linalg.norm(fine['intensity'][region]))
    ratio=float(np.nansum(normal['intensity'][normal_region])/np.nansum(projection['intensity'][normal_region]))
    co=skyrmion_multislice(a,n=cfg.volume_n,dx=cfg.multislice_dx_nm,dz=cfg.multislice_dz_nm,config=cfg,material='Co')
    report=dict(ewald_geometry='passed',unit_texture='passed',coverage_average='passed',
                dz_refinement_relative_L2=error,normal_multislice_over_projection=ratio,
                Co_transmission=co['transmission'],weak_transmission=coarse['transmission'])
    assert error<.15, report
    # Suppression is a design-dependent result, not a condition imposed on a user's custom thickness.
    report['normal_thickness_factor'] = p['normal_thickness_factor']
    (out/'validation.json').write_text(json.dumps(report,indent=2))
    print(report)
    return report


def detector_figure(out=OUT, sample_config=None, detector_config=None):
    """Apply the production detector noise model on an explicitly reciprocal grid."""
    from scattering_calculator.experimental_conditions.detector import detector_hologram
    out.mkdir(parents=True,exist_ok=True)
    cfg = sample_config or FTHConfig()
    detector_cfg = detector_config or DetectorEffectsConfig()
    provenance(out, dict(sample=asdict(cfg), detector=asdict(detector_cfg)))
    c=fth_case(config=cfg); raw=c['images']
    # A declared incident-field-to-counts scale, shared by both helicities.
    scale=detector_cfg.expected_peak_counts/raw.max()
    yy,xx=np.indices(raw.shape[1:]); mid=raw.shape[1]//2
    stop=(xx-mid)**2+(yy-mid)**2<detector_cfg.beamstop_radius_px**2
    measured=[]
    state=np.random.get_state()
    try:
        for i,im in enumerate(raw):
            # The production readout path currently uses the legacy global RNG.
            np.random.seed(detector_cfg.noise_seed+i)
            model=detector_hologram(
                detector_layout=SimpleNamespace(real_space_resolution=c['dx_nm']*1e-9),
                hologram=im*scale,beam_parameters=SimpleNamespace(),real_space_pixel_size=c['dx_nm']*1e-9,
                beamstop=SimpleNamespace(beamstop=stop.astype(float)),
                measurement_config=detector_cfg.measurement,
                detector_params={**detector_cfg.detector, 'noise_seed':detector_cfg.noise_seed+i},
                artifacts_config=detector_cfg.artifacts)
            # Already on a common q grid: deliberately skip flat-detector projection.
            model.hologram_detector=im*scale
            model.add_noise()
            measured.append(model.hologram_exp)
    finally:
        np.random.set_state(state)
    measured=np.array(measured)
    noisy_rec=np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(measured[0]-measured[1])))
    fig,axes=plt.subplots(1,3,figsize=(11,3.5),constrained_layout=True)
    axes[0].imshow(raw[0]*scale,origin='lower',norm=LogNorm(1,detector_cfg.expected_peak_counts)); axes[0].set_title('Expected counts, CR')
    axes[1].imshow(measured[0],origin='lower',norm=LogNorm(1,detector_cfg.expected_peak_counts)); axes[1].set_title('Poisson + readout + defects')
    axes[2].imshow(abs(noisy_rec),origin='lower',extent=c['extent_nm']); axes[2].set(title='Noisy |FTH(CR − CL)|',xlabel='shift (nm)')
    set_sideband_view(axes[2], cfg)
    savefig(fig,'fig08_detector',out)
    np.savez_compressed(out/'detector.npz',expected=raw*scale,measured=measured,beamstop=stop,reconstruction=noisy_rec)
    return measured


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',choices=['all','skyrmion','multislice','fth','detector','validation','fft'],default='all')
    parser.add_argument('--output',type=Path,default=OUT)
    args=parser.parse_args(); start=time.perf_counter()
    provenance(args.output,dict(experiment=args.experiment,geometry=design(),seed=17))
    if args.experiment in ('all','skyrmion'): skyrmion_born(args.output)
    if args.experiment in ('all','multislice'): multislice_figure(args.output)
    if args.experiment in ('all','fft'): fft_volume_comparison(args.output)
    if args.experiment in ('all','fth'): fth_figures(args.output)
    if args.experiment in ('all','detector'): detector_figure(args.output)
    if args.experiment in ('all','validation'): validation(args.output)
    print(f'Elapsed: {time.perf_counter()-start:.1f} s; results: {args.output}')

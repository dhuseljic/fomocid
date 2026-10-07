"""Normal-incidence FTH walkthrough using the production scattering kernels.

Run: MPLBACKEND=Agg python paper/scattering_calculator/workflow.py
The count scale is a declared synthetic budget, not an absolute flux calibration.
"""
from dataclasses import asdict
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch

import experiments as ex
from scattering_calculator.experimental_conditions import detector, light_beam
from scattering_calculator.beam_propagator.detector_propagation import RayleighSommerfeldPropagator
from scattering_calculator.simulation_pipelines.experiment import analyze_jones

OUT = Path(__file__).parent / 'results' / 'fth_workflow'


def project(case, geometry=None):
    """Exit fields -> ideal intensities on physical detector pixels, CR then CL.

    This teaching helper exposes the low-level operators. Fraunhofer retains
    the production FFT coordinate convention; RS uses physical coordinates.
    The two backends have different raw normalizations. Do not compare their
    raw amplitudes as calibrated photon counts.
    """
    g = geometry or ex.fth_experiment().detector
    beam = light_beam.beam_parameters(case['energy_eV'], 'CR', 1., (1., 1.))
    beam.calc_wavevector()
    layout = detector.detector_layout(g.pixel_size, g.shape, g.sample_to_detector_distance,
                                      g.detector_center)
    layout.calc_q_space_coordinates(beam)
    images = []
    for image, field in zip(case['images'], case['exits']):
        if g.detector_propagation_method == 'fraunhofer':
            if g.analyzer_angle is not None:
                amplitude = np.fft.fftshift(np.fft.fft2(field, axes=(0,1), norm='ortho'), axes=(0,1))
                image = abs(analyze_jones(amplitude, g.analyzer_angle))**2
            model = detector.detector_hologram(layout, image, beam,
                                              case['dx_nm'] * 1e-9, None)
            images.append(model.gnomonic_projection(
                use_pixel_footprint=(g.detector_pixel_footprint_samples if g.use_detector_pixel_footprint else 1) > 1,
                pixel_footprint_samples=(g.detector_pixel_footprint_samples if g.use_detector_pixel_footprint else 1),
                ignore_flat_detector_curvature=g.ignore_flat_detector_curvature))
        else:
            power = np.zeros(g.shape)
            offsets = ((np.arange((g.detector_pixel_footprint_samples if g.use_detector_pixel_footprint else 1))+.5)/(g.detector_pixel_footprint_samples if g.use_detector_pixel_footprint else 1)-.5)*g.pixel_size
            for oy in offsets:
                for ox in offsets:
                    op = RayleighSommerfeldPropagator(field.shape[:2], case['dx_nm']*1e-9,
                        beam.wavelength, g.sample_to_detector_distance, layout.detx+ox, layout.dety+oy)
                    amplitude = op.forward(field)
                    if g.analyzer_angle is not None:
                        amplitude = analyze_jones(amplitude, g.analyzer_angle)
                    value = abs(amplitude)**2
                    power += value if value.ndim == 2 else value.sum(axis=-1)
            images.append(power/(g.detector_pixel_footprint_samples if g.use_detector_pixel_footprint else 1)**2 * (g.pixel_size/(case['dx_nm']*1e-9))**2)
    return np.asarray(images), layout


def acquire(ideal, layout, effects=None):
    """Apply one common synthetic photon scale to both helicities, then noise."""
    cfg = ex._detector_parameters(effects)
    if not np.isfinite(ideal).all() or ideal.min() < 0 or ideal.max() <= 0:
        raise ValueError('Expected finite nonnegative intensities with nonzero power')
    scale = cfg.expected_peak_counts / ideal.max()
    expected = ideal * scale
    yy, xx = np.indices(layout.detector_shape)
    cy, cx = layout.detector_center
    stop = (xx-cx)**2+(yy-cy)**2 < cfg.beamstop_radius_px**2
    measured = []
    state = np.random.get_state()
    try:
        for i, image in enumerate(expected):
            # Legacy readout draws use global NumPy state; keep caller state intact.
            np.random.seed(cfg.noise_seed+i)
            model = detector.detector_hologram(layout, image, None, 1., None,
                artifacts_config=cfg.artifacts, measurement_config=cfg.measurement,
                detector_params={**cfg.detector, 'noise_seed': cfg.noise_seed+i})
            model.hologram_detector = image.copy()
            model.add_noise(apply_beamstop_mask=False)
            observed = model.hologram_exp.copy()
            observed[stop] = 0
            measured.append(observed)
    finally:
        np.random.set_state(state)
    return dict(expected=expected, measured=np.asarray(measured), beamstop=stop, scale=scale)


def setup_figure(out=OUT, config=None, geometry=None):
    """Vector schematic, built from the same sample/detector settings as Fig. 2."""
    experiment = ex._fth_experiment(config)
    cfg, g = ex._fth_parameters(experiment), geometry or experiment.detector
    fig = plt.figure(figsize=(13, 7.2), facecolor='white')
    ax = fig.add_axes([.04, .41, .92, .55]); ax.set(xlim=(0, 13), ylim=(0, 5)); ax.axis('off')
    ax.text(0, 4.7, 'a  Normal-incidence Fourier-transform holography', weight='bold', fontsize=15)
    ax.annotate('', (4.0, 2.4), (.5, 2.4), arrowprops=dict(arrowstyle='->', lw=3, color='#247a9b'))
    ax.text(.5, 2.8, f'Coherent X-rays\n{cfg.energy_eV:g} eV · CR / CL', fontsize=12)
    ax.text(2.2, 1.9, 'incident direction +z', fontsize=10)
    for i, (mat, thick) in enumerate(cfg.layers_nm):
        xpos = 4.1+i*.23
        ax.add_patch(Rectangle((xpos, .9), .18, 3, facecolor=plt.cm.cividis(.2+i*.14)))
        ax.add_patch(Rectangle((xpos, 1.55), .18, .1, facecolor='white'))
        if mat == 'Au':
            ax.add_patch(Rectangle((xpos, 2.35), .18, .7, facecolor='white'))
        ax.text(xpos+.09, .7, mat, rotation=65, ha='right', fontsize=9)
    ax.text(4.5, 4.2, 'Patterned multilayer', ha='center', fontsize=12)
    for y in (1.6, 2.7):
        ax.plot([5.2, 10.8], [y, .9], color='#247a9b', alpha=.45)
        ax.plot([5.2, 10.8], [y, 3.8], color='#247a9b', alpha=.45)
    ax.annotate('', (10.4, .35), (5.4, .35), arrowprops=dict(arrowstyle='<->', color='#555'))
    ax.text(7.9, .55, f'free space · L = {g.sample_to_detector_distance*100:g} cm', ha='center', fontsize=11)
    ax.add_patch(Rectangle((10.8, .8), .6, 3.1, facecolor='#dde8ee', edgecolor='#247a9b'))
    ax.text(11.1, 4.2, 'Flat detector', ha='center', fontsize=12)
    ax.text(11.5, 2.4, f'{g.shape[0]} × {g.shape[1]}\n{g.pixel_size*1e6:g} µm pixels', fontsize=10)
    inset = fig.add_axes([.07, .44, .16, .15]); inset.set_aspect('equal')
    inset.set(xlim=(-250, 750), ylim=(-250, 250)); inset.axis('off')
    inset.add_patch(Rectangle((-250,-250),1000,500,facecolor='#e6e8eb'))
    inset.add_patch(Circle((0,0),cfg.object_radius_nm,facecolor='#ba4a64'))
    inset.add_patch(Circle(cfg.reference_xy_nm,cfg.reference_radius_nm,facecolor='white',edgecolor='#333'))
    inset.text(0, -230, 'object', ha='center', fontsize=9)
    inset.text(cfg.reference_xy_nm[0], -110, 'reference', ha='center', fontsize=9)
    inset.text(240, 300, 'Mask front view', ha='center', fontsize=10)
    ax.text(8, 4.2, 'schematic · not to scale', ha='center', fontsize=9, color='#555')
    flow = fig.add_axes([.035, .05, .93, .32]); flow.set(xlim=(-.1, 13.1), ylim=(0, 3)); flow.axis('off')
    flow.text(0, 2.8, 'b  Simulation workflow and user controls', weight='bold', fontsize=15)
    boxes = [
        ('1  Sample + optical data', 'layers · holes · m(x,y,z)\nenergy → $n_0$, $n_c$, $n_l$'),
        ('2  Light–matter interaction', 'Scalar / Jones / Stokes\nlocal transmission per slice'),
        ('3  Within-sample propagation', 'multislice on / off\nangular spectrum · slice size'),
        ('4  Exit wave → detector', 'Fraunhofer / finite-distance RS\ndistance · pitch · pixel footprint'),
        ('5  Acquisition', 'photon budget · beamstop\nnoise · saturation · defects')]
    for i,(title,body) in enumerate(boxes):
        x = i*2.62
        flow.add_patch(FancyBboxPatch((x+.05,.8),2.42,1.5,boxstyle='round,pad=.06',
                                     facecolor='#edf4f7',edgecolor='#247a9b'))
        flow.text(x+1.26,1.96,title,ha='center',fontsize=9,weight='bold')
        flow.text(x+1.26,1.42,body,ha='center',va='center',fontsize=9)
        if i < 4: flow.annotate('',(x+2.64,1.55),(x+2.48,1.55),arrowprops=dict(arrowstyle='->'))
    flow.text(0,.22,'Outputs: complex exit fields → ideal holograms → detector counts; FTH sidebands are a downstream diagnostic.',fontsize=11)
    ex.savefig(fig, 'fig01_setup_workflow', out)


def baseline(out=OUT, config=None, geometry=None, effects=None):
    experiment = ex._fth_experiment(config)
    cfg, g, eff = ex._fth_parameters(experiment), geometry or experiment.detector, ex._detector_parameters(effects or experiment.detector)
    ex.provenance(out, experiment.to_dict())
    record = json.loads((out/'provenance.json').read_text())
    record['source_sha256']['paper/scattering_calculator/workflow.py'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (out/'provenance.json').write_text(json.dumps(record,indent=2)+'\n')
    setup_figure(out, experiment, g)
    case = ex.fth_case(config=experiment)
    ideal, layout = project(case, g)
    acquisition = acquire(ideal, layout, eff)
    np.savez_compressed(out/'baseline.npz', **case, detector_ideal=ideal,
                        **acquisition, detector_qx=layout.detqx, detector_qy=layout.detqy)
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5), constrained_layout=True)
    axes[0].imshow(case['mz']*case['object_hole'], origin='lower', extent=case['extent_nm'], cmap='RdBu_r',vmin=-1,vmax=1)
    axes[0].contour(case['reference_hole'],levels=[.5],extent=case['extent_nm'],colors='black')
    axes[0].set(title='Sample: Co magnetization',xlabel='x (nm)',ylabel='y (nm)')
    axes[1].imshow(np.sum(abs(case['exits'][0])**2,axis=-1),origin='lower',extent=case['extent_nm'],cmap='magma')
    axes[1].set(title='CR exit intensity',xlabel='x (nm)')
    norm=LogNorm(1,eff.expected_peak_counts)
    for ax,image,title in zip(axes[2:],(acquisition['expected'][0],acquisition['measured'][0]),('Ideal detector, CR','Detector with artifacts, CR')):
        ax.imshow(image,origin='lower',norm=norm,cmap='magma'); ax.set(title=title,xlabel='detector column')
    ex.savefig(fig,'fig02_fth_chain',out)
    return case, ideal, acquisition


def spectral(out=OUT, config=None, geometry=None):
    """Both the common-q reconstruction cube and a fixed-camera energy series."""
    experiment = ex._fth_experiment(config)
    cfg, g = ex._fth_parameters(experiment), geometry or experiment.detector
    if any(not 770 <= e <= 805 for e in cfg.energies_eV):
        raise ValueError('This Co example needs energies within the magnetic-table window, 770–805 eV')
    folder = out/'spectral'; folder.mkdir(parents=True,exist_ok=True)
    # Shared existing implementation generates optical curves, mode comparison,
    # and a common-q complex reconstruction cube with raw arrays.
    metrics = ex.fth_figures(folder, config=experiment)
    frames, qx, qy = [], [], []
    for energy in cfg.energies_eV:
        case = ex.fth_case(energy=energy,config=experiment)
        image, layout = project(case,g)
        frames.append(image); qx.append(layout.detqx); qy.append(layout.detqy)
    np.savez_compressed(folder/'fixed_detector_series.npz', energies_eV=cfg.energies_eV,
                        images=frames,qx=qx,qy=qy)
    (folder/'geometry.json').write_text(json.dumps(asdict(g),indent=2)+'\n')
    for old,new in [('fig05_fth','fig03_fth_reconstruction'),('fig06_modes','fig05_interaction_modes'),('fig07_hyperspectral','fig04_hyperspectral')]:
        for suffix in ('png','pdf'):
            (out/f'{new}.{suffix}').write_bytes((folder/f'{old}.{suffix}').read_bytes())
    return metrics


def validate(out=OUT, config=None):
    """FTH checks, with reported refinement differences rather than an accuracy claim."""
    experiment = ex._fth_experiment(config)
    cfg = ex._fth_parameters(experiment)
    cases = {m: ex.fth_case(mode=m,config=experiment) for m in ('Scalar','Jones','Stokes')}
    ref = cases['Jones']; den = np.linalg.norm(ref['difference'])
    metrics = {m:float(np.linalg.norm(c['difference']-ref['difference'])/den) for m,c in cases.items()}
    np.testing.assert_allclose(cases['Stokes']['difference'],ref['difference'],rtol=1e-10,atol=1e-12)
    refined = ex.fth_case(config=experiment.with_changes(sample={'max_slice_thickness':cfg.max_slice_nm*1e-9/2}))
    metrics['slice_halving_relative_L2'] = float(np.linalg.norm(refined['difference']-ref['difference'])/den)
    projected = ex.fth_case(config=experiment.with_changes(propagation={'propagator_config':{**experiment.propagation.propagator_config,'propagate':False}}))
    metrics['projection_control_relative_L2'] = float(np.linalg.norm(projected['difference']-ref['difference'])/den)
    ideal,layout = project(ref,experiment.detector)
    a,b = acquire(ideal,layout,experiment.detector),acquire(ideal,layout,experiment.detector)
    np.testing.assert_array_equal(a['measured'],b['measured'])
    assert np.isfinite(ideal).all() and ideal.min() >= 0
    metrics['seeded_acquisition_repeatable'] = True
    metrics['detector_finite_nonnegative'] = True
    out.mkdir(parents=True,exist_ok=True)
    (out/'validation.json').write_text(json.dumps(metrics,indent=2)+'\n')
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', choices=('all','baseline','spectral','validation'),default='all')
    parser.add_argument('--output',type=Path,default=OUT)
    args = parser.parse_args()
    if args.experiment in ('all','baseline'): baseline(args.output)
    if args.experiment in ('all','spectral'): spectral(args.output)
    if args.experiment in ('all','validation'): print(json.dumps(validate(args.output),indent=2))

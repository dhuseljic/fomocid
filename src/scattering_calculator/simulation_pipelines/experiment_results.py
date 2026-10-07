"""Run configured experiments, stream HDF5 results, and plot saved observables."""
from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from copy import deepcopy
from collections import Counter
from datetime import datetime, timezone
import itertools
import hashlib
import json
import os
import platform
from pathlib import Path
import tempfile
import h5py
import numpy as np

from .experiment import ExperimentConfig, ScatteringExperiment

FORMAT = 'scattering_calculator.experiment'
SCHEMA_VERSION = 1
OBSERVABLES = frozenset(('exit_wave','exit_wave_for_farfield','exit_stokes','fft','fft_intensity','detector_ideal',
    'detector_measured','magnetization','sample_mask','illumination','reconstruction'))


def _json_default(value):
    """Encode NumPy values, paths and callback identifiers for provenance.

    Parameters
    ----------
    value : object
        Value not handled by the standard JSON encoder.

    Returns
    -------
    object
        JSON-compatible value; unsupported types raise TypeError.
    """
    if isinstance(value, Path): return str(value)
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if callable(value): return {'callback':f'{value.__module__}.{value.__qualname__}'}
    raise TypeError(f'Cannot record experiment value {type(value).__name__} in JSON')


def _scan_parameters(config):
    """Find parameter alternatives, preserving structural arrays and output lists."""
    axes = {}
    structural = {'shape', 'coherence_length', 'detector_center', 'center', 'alpha_beam',
                  'reference_xy', 'magnetic_materials', 'cosmic_ray_length_range',
                  'cosmic_ray_aspect_ratio_range', 'spectral_components', 'input_stokes', 'aperture_thicknesses',
                  'aperture_layer_names', 'sample_shape', 'aperture_shape', 'bs_center',
                  'ellipticity', 'roughness_modes', 'direction', 'magnetization_direction'}
    def visit(value, path):
        if path and path[0] in ('outputs', 'analysis'): return
        if path and (path[-1] in ('energies_eV', 'polarizations') or
                     path[-1] in structural or path[-1].startswith('apertures_')): return
        if isinstance(value, list):
            if not value: raise ValueError(f"Empty scan axis: {'.'.join(path)}")
            axes['.'.join(path)] = value
        elif is_dataclass(value):
            for item in fields(value): visit(getattr(value, item.name), (*path, item.name))
        elif isinstance(value, dict):
            for name, item in value.items(): visit(item, (*path, name))
    visit(config, ())
    return axes


def _select_parameters(config, selections):
    """Resolve scalar selections in an independent experiment declaration."""
    current = deepcopy(config)
    for path, value in selections.items():
        parts = path.split('.')
        target = current
        for name in parts[:-1]: target = target[name] if isinstance(target, dict) else getattr(target, name)
        if isinstance(target, dict): target[parts[-1]] = deepcopy(value)
        else: setattr(target, parts[-1], deepcopy(value))
    return current


def _sample_stage_key(config):
    """Identify physical upstream settings, excluding detector/acquisition stages."""
    upstream = config.to_dict()
    for section in ('detector','beamstop','outputs','analysis','energies_eV'):
        upstream.pop(section, None)
    if upstream['propagation']['propagator_method'] == 'Stokes':
        upstream['propagation']['propagator_config']['calculate_farfield'] = True
    return json.dumps(upstream,default=_json_default,sort_keys=True)


def _coherent_fields(run, *, farfield=False):
    """Get coherent channels, keeping mixed-state modes mutually incoherent.

    Parameters
    ----------
    run : ScatteringExperiment
        Propagated run.
    farfield : bool
        Use the padded exit plane when available.

    Returns
    -------
    ndarray
        Scalar, Jones, or Jones fields with a trailing incoherent-mode axis.
    """
    if hasattr(run, "spectral_runs"):
        return np.stack([_coherent_fields(component, farfield=farfield)
                         for component in run.spectral_runs], axis=-1)
    wave = run.wavefront
    attribute = 'exit_wave_for_farfield' if farfield else 'exit_wave'
    if run.config.propagation.propagator_method == 'Stokes':
        return np.stack([getattr(mode,attribute) for mode in wave._coherent_modes],axis=-1)
    field = getattr(wave,attribute,None)
    return wave.exit_wave if field is None else field


def _power(field):
    """Sum polarization and incoherent-mode powers at each spatial pixel.

    Parameters
    ----------
    field : ndarray
        Scalar or coherent channel amplitudes with spatial axes first.

    Returns
    -------
    ndarray
        Two-dimensional intensity.
    """
    value = abs(field)**2
    return value.sum(axis=tuple(range(2,value.ndim))) if value.ndim > 2 else value


@dataclass(frozen=True)
class ExperimentResults:
    """A reloadable result file; large observables are read only on request."""
    path: Path
    config: dict
    energies_eV: tuple[float, ...]
    polarizations: tuple[str, ...]
    saved: tuple[str, ...]
    parameter_axes: dict

    def read(self, name, *, energy_index=0, polarization_index=0, scan_index=0):
        """Load one saved observable or coordinate from a selected exposure.

        Parameters
        ----------
        name : str
            Observable name or coordinate path, e.g. coordinates/qx.
        energy_index, polarization_index : int
            Positions in the saved scan axes, including repeated scan points.

        Returns
        -------
        ndarray
            Saved array, independent of the file handle.
        """
        if not 0 <= energy_index < len(self.energies_eV) or not 0 <= polarization_index < len(self.polarizations):
            raise IndexError('Exposure index outside the saved scan axes')
        combinations = int(np.prod([len(v) for v in self.parameter_axes.values()]))
        if not 0 <= scan_index < combinations: raise IndexError("Parameter scan index outside saved axes")
        index = (scan_index*len(self.energies_eV)+energy_index)*len(self.polarizations)+polarization_index
        with h5py.File(self.path,'r') as file:
            key=f'runs/{index:06d}/{name}'
            if key not in file: raise KeyError(f'{name!r} was not saved; requested observables: {self.saved}')
            return file[key][()]

    def stack(self, name, *, scan_index=0):
        """Load an observable for the full energy-by-polarization scan.

        Parameters
        ----------
        name : str
            Saved observable or coordinate path.

        Returns
        -------
        ndarray
            Axes are energy, polarization, followed by the observable dimensions.
        """
        return np.stack([np.stack([self.read(name,energy_index=i,polarization_index=j,scan_index=scan_index)
            for j in range(len(self.polarizations))]) for i in range(len(self.energies_eV))])


def load_results(path):
    """Open completed experiment metadata without loading large field arrays.

    Parameters
    ----------
    path : str or Path
        HDF5 file created by simulate_experiment.

    Returns
    -------
    ExperimentResults
        Reader usable for plotting and analysis without resimulation.
    """
    path=Path(path).resolve()
    with h5py.File(path,'r') as file:
        if file.attrs.get('format') != FORMAT or file.attrs.get('schema_version') != SCHEMA_VERSION:
            raise ValueError('Not a supported scattering-calculator experiment result')
        if file.attrs.get('status') != 'complete': raise ValueError('Experiment file is incomplete')
        config=json.loads(file['config_json'].asstr()[()])
        energies=tuple(float(e) for e in file['axes/energies_eV'][()])
        states=tuple(file['axes/polarizations'].asstr()[()])
        saved=tuple(file['axes/saved_observables'].asstr()[()])
        parameters=json.loads(file['axes'].attrs.get('parameter_axes_json','{}'))
    return ExperimentResults(path,config,energies,states,saved,parameters)


def simulate_experiment(config, *, magnetization=None, mask=None):
    """Run one exposure or a scan and atomically save configured HDF5 outputs.

    Parameters
    ----------
    config : ExperimentConfig
        Complete declaration including illumination scan lists and outputs.
    magnetization, mask : ndarray or None
        Optional custom specimen arrays passed to the standard setup stage.

    Returns
    -------
    ExperimentResults
        Reloadable saved results. No detector argument or plotting is required.
    """
    if not isinstance(config,ExperimentConfig): raise TypeError('Use ExperimentConfig')
    config=deepcopy(config)
    if isinstance(config.xray.energy, list):
        if config.illumination.energies_eV is not None or config.energies_eV is not None:
            raise ValueError('Declare energy alternatives once: source or illumination')
        config.illumination.energies_eV = tuple(config.xray.energy)
        if not config.xray.energy: raise ValueError('Empty energy scan')
        config.xray.energy = config.xray.energy[0]
    if isinstance(config.xray.pol, list):
        if config.illumination.polarizations is not None:
            raise ValueError('Declare polarization alternatives once: source or illumination')
        config.illumination.polarizations = tuple(config.xray.pol)
        if not config.xray.pol: raise ValueError('Empty polarization scan')
        config.xray.pol = config.xray.pol[0]
    parameter_axes = _scan_parameters(config)
    reference_config = _select_parameters(config, {k:v[0] for k,v in parameter_axes.items()})
    # A detector-derived grid is resolved once at the declared reference energy.
    # Changing energy must change optical data/q, never the physical specimen grid.
    if config.simulation.other_config.get('grid_mode')=='detector':
        factor=config.simulation.other_config.get('oversampling',2)
        if not isinstance(factor,int) or factor<1: raise ValueError('Grid oversampling must be a positive integer')
        camera=deepcopy(reference_config.detector);camera.setup()
        reference=deepcopy(reference_config.xray).setup()
        config.simulation.shape=tuple(factor*n for n in camera.shape)
        config.simulation.real_space_pixel_size=camera.calc_realspace_resolution(reference)/factor
        config.simulation.other_config.update(grid_mode='explicit',sampling_reference_energy_eV=config.xray.energy)
    if magnetization is None and any(method in (
        'wavy_stripe_pattern','binary_labyrinth_pattern','skyrmion_pattern','disordered_skyrmion_lattice_pattern')
        for method in parameter_axes.get('magnetic_pattern.pattern_type_method',
                                        [config.magnetic_pattern.pattern_type_method])):
        if config.magnetic_pattern.pattern_config.get('seed') is None:
            config.magnetic_pattern.pattern_config['seed']=int(np.random.default_rng().integers(0,2**31-1))
    energies,states=reference_config.scan_axes()
    saved=tuple(config.outputs.save)
    if len(saved)!=len(set(saved)) or set(saved)-OBSERVABLES:
        raise ValueError(f'Choose unique output names from {sorted(OBSERVABLES)}')
    if 'exit_stokes' in saved and any(method != 'Stokes' for method in
        parameter_axes.get('propagation.propagator_method',[config.propagation.propagator_method])):
        raise ValueError('exit_stokes output requires Stokes propagation')
    if config.outputs.compression not in (None,'gzip','lzf'):
        raise ValueError('HDF5 compression must be None, gzip or lzf')
    if 'detector_measured' in saved:
        seed=config.detector.detector_params.get('noise_seed')
        if isinstance(seed, list): seed = seed[0]
        if seed is None:
            seed=int(np.random.default_rng().integers(0,2**31-1))
            config.detector.detector_params['noise_seed']=seed
        persistent=np.random.default_rng(int(seed))
        names=['camera_seed']
        if np.any(np.asarray(config.detector.artifacts_config.get('sigma_photon',0.))>0):
            names.extend(('photon_kernel_seed','photon_class_seed'))
        for name in names:
            if config.detector.artifacts_config.get(name) is None:
                config.detector.artifacts_config[name]=int(persistent.integers(0,2**31-1))
    record=json.dumps(config.to_dict(),default=_json_default,sort_keys=True)
    path=Path(config.outputs.path).resolve()
    if path.is_dir(): raise IsADirectoryError(path)
    if path.exists() and not config.outputs.overwrite: raise FileExistsError(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    descriptor,temporary=tempfile.mkstemp(prefix='.'+path.name+'-',suffix='.tmp',dir=path.parent)
    os.close(descriptor)
    try:
        with h5py.File(temporary,'w') as file:
            file.attrs.update(format=FORMAT,schema_version=SCHEMA_VERSION,status='running',
                created_utc=datetime.now(timezone.utc).isoformat(),numpy_version=np.__version__,
                python_version=platform.python_version(),h5py_version=h5py.__version__)
            file.create_dataset('config_json',data=record,dtype=h5py.string_dtype())
            inputs=file.create_group('inputs')
            for name,data in (('magnetization',magnetization),('sample_mask',mask)):
                if data is not None: inputs.create_dataset(name,data=data,compression=config.outputs.compression)
            axes=file.create_group('axes')
            axes.attrs['parameter_axes_json']=json.dumps(parameter_axes,default=_json_default)
            axes.create_dataset('energies_eV',data=energies)
            axes.create_dataset('polarizations',data=states,dtype=h5py.string_dtype())
            axes.create_dataset('saved_observables',data=saved,dtype=h5py.string_dtype())
            provenance=file.create_group('provenance')
            package=Path(__file__).resolve().parents[1]
            for source in sorted(package.rglob('*.py')):
                provenance.attrs[str(source.relative_to(package))]=hashlib.sha256(source.read_bytes()).hexdigest()
            stage_cache = {}
            remaining = Counter()
            for values in itertools.product(*parameter_axes.values()):
                selected = _select_parameters(config, dict(zip(parameter_axes, values)))
                for energy, state in itertools.product(energies, states):
                    remaining[_sample_stage_key(selected.with_changes(xray={'energy':energy,'pol':state}))] += 1
            combinations = itertools.product(*parameter_axes.values())
            for scan_index, values in enumerate(combinations):
                selections = dict(zip(parameter_axes, values))
                selected = _select_parameters(config, selections)
                for i,energy in enumerate(energies):
                    for j,state in enumerate(states):
                        current=selected.with_changes(xray={'energy':energy,'pol':state})
                        seed=current.detector.detector_params.get('noise_seed')
                        if seed is not None:
                            current.detector.detector_params['noise_seed']=int(seed)+(scan_index*len(energies)+i)*len(states)+j
                        # Stokes requires coherent far fields for its public intensity.
                        if current.propagation.propagator_method=='Stokes':
                            current.propagation.propagator_config['calculate_farfield']=True
                        key = _sample_stage_key(current)
                        reused = key in stage_cache
                        if reused:
                            run = stage_cache[key]
                            run.config.detector = deepcopy(current.detector)
                            run.config.beamstop = deepcopy(current.beamstop)
                            run.config.detector.beamstop_config = run.config.beamstop
                            run.config.detector.setup()
                        else:
                            run=ScatteringExperiment(current).setup(magnetization=magnetization,mask=mask)
                            run.propagate()
                            if remaining[key] > 1: stage_cache[key] = run
                        remaining[key] -= 1
                        if remaining[key] == 0: stage_cache.pop(key, None)
                        ideal=run.detect().copy()
                        if not np.isfinite(ideal).all() or np.any(ideal<0):
                            raise ValueError('Detector projection produced invalid ideal intensity')
                        group=file.create_group(f'runs/{(scan_index*len(energies)+i)*len(states)+j:06d}')
                        group.attrs.update(energy_eV=energy,polarization=state,energy_index=i,
                            polarization_index=j,representation=current.propagation.propagator_method)
                        if seed is not None: group.attrs['noise_seed']=current.detector.detector_params['noise_seed']
                        group.attrs.update(scan_index=scan_index, selections_json=json.dumps(selections,default=_json_default),
                            sample_propagation_reused=reused, projection_energy_eV=(current.detector.projection_energy or
                                (run.spectral_runs[0].config.xray.energy if hasattr(run,"spectral_runs") else energy)))
                        coords=group.create_group('coordinates')
                        c=run.config; layout=c.detector.detector_layout
                        for name,data in [('sample_x',c.simulation.xgrid[0]),('sample_y',c.simulation.ygrid[:,0]),
                            ('detector_x',layout.detx),('detector_y',layout.dety),('qx',layout.detqx),('qy',layout.detqy)]:
                            coords.create_dataset(name,data=data)
                        if hasattr(run, 'spectral_runs'):
                            components = group.create_group('spectral_components')
                            for component_index, (component, fraction) in enumerate(zip(run.spectral_runs, run.spectral_weights)):
                                part = components.create_group(f'{component_index:06d}')
                                part.attrs.update(energy_eV=component.config.xray.energy,
                                    polarization=component.config.xray.pol, photon_flux_fraction=fraction)
                                part.create_dataset('qx', data=component.config.detector.detector_layout.detqx)
                                part.create_dataset('qy', data=component.config.detector.detector_layout.detqy)
                                part.create_dataset('refractive_indices', data=component.sample.layer_refractive_indices)
                                for observable in ('exit_wave','exit_wave_for_farfield'):
                                    if observable in saved:
                                        part.create_dataset(observable, data=_coherent_fields(component,farfield=observable.endswith('for_farfield')),
                                                            compression=config.outputs.compression)
                                if 'detector_ideal' in saved:
                                    # Save component images separately before the combined detector image.
                                    part.create_dataset('detector_ideal', data=component.spectral_ideal,
                                                        compression=config.outputs.compression)
                        material_sample = run.spectral_runs[0].sample if hasattr(run,'spectral_runs') else run.sample
                        materials=group.create_group('materials')
                        materials.create_dataset('layer_names',data=material_sample.layer_names,dtype=h5py.string_dtype())
                        materials.create_dataset('thickness_m',data=material_sample.layer_thicknesses)
                        materials.create_dataset('refractive_indices',data=material_sample.layer_refractive_indices)
                        fourier=None
                        if set(saved)&{'fft','fft_intensity','reconstruction'}:
                            field=_coherent_fields(run,farfield=True)
                            fourier=np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field,axes=(0,1)),axes=(0,1)),axes=(0,1))
                            fft_power=_power(fourier)/np.prod(fourier.shape[:2])
                            coords.create_dataset('fft_qx',data=2*np.pi*np.fft.fftshift(np.fft.fftfreq(field.shape[1],d=c.simulation.real_space_pixel_size)))
                            coords.create_dataset('fft_qy',data=2*np.pi*np.fft.fftshift(np.fft.fftfreq(field.shape[0],d=c.simulation.real_space_pixel_size)))
                        for name in saved:
                            if name=='exit_wave': data=_coherent_fields(run)
                            elif name=='exit_wave_for_farfield': data=_coherent_fields(run,farfield=True)
                            elif name=='exit_stokes':
                                data=(sum(part.wavefront.exit_stokes for part in run.spectral_runs)
                                      if hasattr(run,'spectral_runs') else run.wavefront.exit_stokes)
                            elif name=='fft': data=fourier
                            elif name=='fft_intensity': data=fft_power
                            elif name=='reconstruction': data=np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(fft_power)))
                            elif name=='detector_ideal': data=ideal
                            elif name=='detector_measured':
                                previous=np.random.get_state() if seed is not None else None
                                try:
                                    if seed is not None: np.random.seed(current.detector.detector_params['noise_seed'])
                                    data=run.config.detector.return_detected_hologram()
                                finally:
                                    if previous is not None: np.random.set_state(previous)
                            elif name=='magnetization': data=run.sample.magnetization
                            elif name=='sample_mask': data=run.sample.mask
                            elif name=='illumination':
                                data=(np.stack([mode.E_in for mode in run.wavefront._coherent_modes],axis=-1)
                                      if current.propagation.propagator_method=='Stokes' else run.wavefront.E_in)
                            if name=='illumination' and hasattr(run,'spectral_runs'):
                                data=np.stack([np.stack([mode.E_in for mode in part.wavefront._coherent_modes],axis=-1)
                                    if current.propagation.propagator_method=='Stokes' else part.wavefront.E_in
                                    for part in run.spectral_runs],axis=-1)
                            if not np.isfinite(data).all(): raise ValueError(f'Nonfinite observable: {name}')
                            dataset=group.create_dataset(name,data=data,compression=config.outputs.compression)
                            if name in ('exit_wave','exit_wave_for_farfield','fft','illumination'):
                                dataset.attrs['channel_axes']='scalar' if data.ndim==2 else ('Ex,Ey; incoherent_modes' if data.ndim==4 else 'Ex,Ey')
                            if hasattr(run,'spectral_runs') and name in ('exit_wave','exit_wave_for_farfield','fft','illumination'):
                                dataset.attrs['channel_axes']=('scalar; spectral_components' if current.propagation.propagator_method=='Scalar'
                                    else ('Ex,Ey; incoherent_polarization_modes; spectral_components'
                                          if current.propagation.propagator_method=='Stokes' else 'Ex,Ey; spectral_components'))
                                dataset.attrs['spectral_axis']=-1
                                dataset.attrs['spectral_combination']='mutually incoherent; sum squared amplitudes'
                            if name=='fft': dataset.attrs['normalization']='raw forward FFT'
                            if name=='fft_intensity': dataset.attrs['normalization']='Parseval: sum equals padded exit-field power; before analyzer'
                            if name=='detector_measured': dataset.attrs['units']='detector counts (frame average)'
                            if name=='detector_ideal': dataset.attrs['units']='incident-rate intensity after optical analyzer; before sensor response and beamstop'
            file.attrs['status']='complete'
        if config.outputs.overwrite: os.replace(temporary,path)
        else:
            os.link(temporary,path)  # exclusive publication preserves an existing file
            os.unlink(temporary)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)
    return load_results(path)


def plot_results(results, *, outputs=None, energy_index=0, polarization_index=0, scan_index=0, save_path=None):
    """Plot selected saved arrays without running any simulation stages.

    Parameters
    ----------
    results : ExperimentResults or str or Path
        Loaded results or their HDF5 filename.
    outputs : sequence of str or None
        Panels to display; defaults to available configured summary panels.
    energy_index, polarization_index : int
        Exposure to display.
    save_path : str or Path or None
        Optional summary figure destination.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing saved intensities with physical coordinate labels.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    if not isinstance(results,ExperimentResults): results=load_results(results)
    selected=tuple(outputs) if outputs is not None else tuple(name for name in results.config['outputs']['plots'] if name in results.saved)
    if not selected: raise ValueError('No saved outputs selected for plotting')
    fig,axes=plt.subplots(1,len(selected),figsize=(4*len(selected),3.6),squeeze=False,constrained_layout=True)
    for ax,name in zip(axes[0],selected):
        value=results.read(name,energy_index=energy_index,polarization_index=polarization_index,scan_index=scan_index)
        if name=='reconstruction': value=abs(value)
        elif np.iscomplexobj(value): value=_power(value)
        elif value.ndim!=2: raise ValueError(f'{name} is a volume or vector observable; select a slice explicitly for analysis')
        kwargs={}
        if name.startswith('detector'):
            x=results.read('coordinates/detector_x',energy_index=energy_index,polarization_index=polarization_index,scan_index=scan_index)
            y=results.read('coordinates/detector_y',energy_index=energy_index,polarization_index=polarization_index,scan_index=scan_index)
            pitch=results.config['detector']['pixel_size']
            kwargs['extent']=((x.min()-pitch/2)*1e3,(x.max()+pitch/2)*1e3,(y.min()-pitch/2)*1e3,(y.max()+pitch/2)*1e3)
            ax.set(xlabel='detector x (mm)',ylabel='detector y (mm)')
        elif name in ('exit_wave','illumination'):
            x=results.read('coordinates/sample_x',energy_index=energy_index,polarization_index=polarization_index,scan_index=scan_index)
            y=results.read('coordinates/sample_y',energy_index=energy_index,polarization_index=polarization_index,scan_index=scan_index)
            pitch=x[1]-x[0] if len(x)>1 else results.config['simulation']['real_space_pixel_size']
            kwargs['extent']=((x.min()-pitch/2)*1e9,(x.max()+pitch/2)*1e9,(y.min()-pitch/2)*1e9,(y.max()+pitch/2)*1e9)
            ax.set(xlabel='sample x (nm)',ylabel='sample y (nm)')
        elif name.startswith('fft'):
            x=results.read('coordinates/fft_qx',energy_index=energy_index,polarization_index=polarization_index,scan_index=scan_index)
            y=results.read('coordinates/fft_qy',energy_index=energy_index,polarization_index=polarization_index,scan_index=scan_index)
            dx=x[1]-x[0] if len(x)>1 else 0.;dy=y[1]-y[0] if len(y)>1 else 0.
            kwargs['extent']=((x.min()-dx/2)*1e-9,(x.max()+dx/2)*1e-9,(y.min()-dy/2)*1e-9,(y.max()+dy/2)*1e-9)
            ax.set(xlabel='qx (rad/nm)',ylabel='qy (rad/nm)')
        else: ax.set(xlabel='x pixel',ylabel='y pixel')
        if name.startswith(('fft','detector')) and value.max()>0:
            positive=value[value>0];low=max(float(positive.min()),float(value.max())*1e-8)
            if low<float(value.max()):
                kwargs['norm']=LogNorm(low,float(value.max()))
                value=np.maximum(value,low)  # display floor; retain signed raw sensor data in HDF5
        image=ax.imshow(value,origin='lower',cmap='magma',**kwargs)
        ax.set_title(name.replace('_',' '));fig.colorbar(image,ax=ax,shrink=.8)
    fig.suptitle(f'{results.energies_eV[energy_index]:g} eV · {results.polarizations[polarization_index]}')
    if save_path is not None:
        path=Path(save_path);path.parent.mkdir(parents=True,exist_ok=True);fig.savefig(path,dpi=160)
    return fig


def run_experiment(config, **specimen_arrays):
    """Run, save, reload and render the configured summary in one command.

    Parameters
    ----------
    config : ExperimentConfig
        Physical settings, saved observables and summary-plot settings.
    specimen_arrays : dict
        Optional custom arrays forwarded to simulate_experiment.

    Returns
    -------
    tuple
        Saved ExperimentResults and summary figure, or None if plots are disabled.
    """
    results=simulate_experiment(config,**specimen_arrays)
    fig=None
    if config.outputs.plots:
        destination=config.outputs.figure_path or str(results.path.with_suffix('.png'))
        fig=plot_results(results,save_path=destination)
    return results,fig

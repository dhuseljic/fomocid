"""Single-experiment desktop interface using the canonical simulation backend.

Launch with ``python -m scattering_calculator.gui.single_experiment``.
"""
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import dataclass, fields
import ast
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import tempfile
import traceback

import h5py
import numpy as np

from scattering_calculator.simulation_pipelines.pipelines.hologram_pipeline import (
    HologramPipeline, HologramPipelineConfig, HologramPipelineRanges,
)
from scattering_calculator.simulation_pipelines.recipe_sampling import sample_multilayer_recipe


def default_config():
    """Return a resolved, modest-sized example with a complete holography mask."""
    return HologramPipelineConfig(
        recipe='[Au(70.0)Cr(17.5)]x7/SiN(200.0)/Co(10.0)Pt(5.0)Al(5.0)',
        detector_shape=(256, 256), detector_distance=.04, xray_energy=778.,
        xray_photon_flux=1e8, illumination_fwhm=3e-6,
        propagator_method='Scalar', beamstop_method=None,
        aperture_radii=[300e-9, 30e-9, 30e-9],
        aperture_centers=[(0., 0.), (1e-6, 0.), (0., 1e-6)],
        detector_params=dict(readout_noise_average=50., readout_noise_sigma=3.,
            detector_threshold=64000., counts_per_photon=100., quantum_efficiency=1.,
            nonlinear_response=None),
        random_seed=1, pattern_config=dict(
            characteristic_length=160e-9, sigma=5e-9, target_magnetization=None,
            alignment=0., direction=0., disorder=0., disorder_length=0.,
            initial_state='random', initial_order=0., initial_domain_count=1,
            worm_length=400e-9, worm_waviness=30e-9, wall_stiffness=2.,
            n_steps=100, time_step=.1, convergence_tolerance=1e-4,
            length_tolerance=.15, max_length_iterations=1, seed=1),
    )


def parse_value(text, template):
    """Parse editor input as literals; preserve plain string-valued controls."""
    if isinstance(template, str):
        return text
    try:
        value = ast.literal_eval(text)
    except (ValueError, SyntaxError) as error:
        if template is None and text.strip() and not any(c in text for c in '[]{}()'):
            return text.strip()
        raise ValueError('Use a Python literal, such as 0.04, None, or (256, 256)') from error
    if isinstance(template, bool) and not isinstance(value, bool):
        raise ValueError('Expected True or False')
    if isinstance(template, int) and not isinstance(template, bool):
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError('Expected an integer')
    elif isinstance(template, float):
        if not isinstance(value, (float, int)) or isinstance(value, bool):
            raise ValueError('Expected a number')
        value = float(value)
    elif isinstance(template, (tuple, list, dict)) and not isinstance(value, type(template)):
        raise ValueError(f'Expected a {type(template).__name__}')
    return value


@dataclass
class SaturatingResponse:
    """Optional detector compression; strength is the half-response ADU scale."""
    scale: float

    def __call__(self, linear_adu):
        return linear_adu / (1 + linear_adu / self.scale)


def simulate_to_file(config, output_path):
    """Run exactly the selected experiment without resampling or coarsening."""
    HologramPipeline(config=config, ranges=HologramPipelineRanges(),
        output_path=output_path, n_samples=1, verbose=True, max_sampling_attempts=1, detailed_logging=True).run()
    with h5py.File(output_path, 'a') as h5:
        if '00000' not in h5:
            raise ValueError('The selected grid exceeds the allocation limit. Choose a shorter '
                             'distance, wider domain walls, smaller detector, or larger grid limit.')
        gui_metadata = h5['00000/metadata'].create_group('gui')
        response = config.detector_params.get('nonlinear_response')
        gui_metadata.create_dataset('response_preset',
            data='saturating' if isinstance(response, SaturatingResponse) else 'linear')
        if isinstance(response, SaturatingResponse):
            gui_metadata.create_dataset('response_scale_adu', data=response.scale)


def worker(config, output_path, log_path, messages):
    """Execute physics in an isolated process and report completion to the UI."""
    try:
        with open(log_path, 'w', buffering=1) as log, redirect_stdout(log), redirect_stderr(log):
            simulate_to_file(config, output_path)
        messages.put(('done', str(output_path)))
    except Exception:
        messages.put(('error', traceback.format_exc()))


def read_result(path):
    """Load only the four view arrays and saved state information."""
    with h5py.File(path, 'r') as h5:
        group = h5['00000']
        result = {'magnetic': group['magnetic_pattern_oh'][()]}
        for pol in ('CR', 'CL'):
            result[pol] = {source: np.squeeze(group[f'{pol}/{source}'][()])
                          for source in ('exit_wave', 'ideal', 'detected')}
        meta = group['metadata']
        result['recipe'] = meta['sample/recipe'].asstr()[()]
        result['classification'] = json.loads(meta['sample/magnetic_pattern/classification_json'].asstr()[()])
        result['measurements'] = json.loads(meta['sample/magnetic_pattern/measurements_json'].asstr()[()])
        result['pitch'] = float(meta['sample/real_space_pixel_size'][()])
        return result


def polarization_view(cr, cl, mode):
    """Select helicity, subtract fields or divide with undefined pixels masked."""
    if mode == 'CR':
        return cr
    if mode == 'CL':
        return cl
    if mode in ('CL−CR', 'CL-CR'):
        return cl - cr
    if mode in ('CR−CL', 'CR-CL'):  # Compatibility with existing saved UI choices.
        return cr - cl
    if mode == 'CL/CR':
        cr, cl = np.asarray(cr), np.asarray(cl)
        dtype = np.result_type(cr.dtype, cl.dtype, np.float64)
        ratio = np.full(np.broadcast_shapes(cr.shape, cl.shape), np.nan, dtype=dtype)
        valid = (cr != 0) & np.isfinite(cr) & np.isfinite(cl)
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            np.divide(cl, cr, out=ratio, where=valid)
        ratio[~np.isfinite(ratio)] = np.nan
        return ratio
    raise ValueError(f'Unknown polarization view: {mode}')


SECTIONS = (
    'Magnetic pattern', 'Sample structure and grid', 'Holography mask',
    'X-rays and illumination', 'Detector geometry', 'Acquisition and sensor response',
    'Beamstop', 'Propagation',
)


def aperture_preview(config, max_axis=256):
    """Return depth-weighted hole fraction on a small physical preview grid."""
    from scattering_calculator.sample_generator.structures import parse_recipe
    from scattering_calculator.simulation_pipelines.simulation_configuration import (
        XRayConfig, DetectorConfig, FrontApertureConfig,
    )
    layers = parse_recipe(config.recipe).layers
    thicknesses = np.array([layer.thickness for layer in layers])
    names = [layer.material for layer in layers]
    membrane = next((i for i, name in enumerate(names)
                     if FrontApertureConfig._is_silicon_nitride_layer(name)), len(names))
    xray = XRayConfig(energy=config.xray_energy, photon_flux=config.xray_photon_flux)
    xray.setup()
    detector = DetectorConfig(shape=config.detector_shape, pixel_size=config.detector_pixel_size,
        sample_to_detector_distance=config.detector_distance,
        detector_center=config.detector_center or tuple(np.array(config.detector_shape)//2))
    detector.setup()
    resolution = detector.calc_realspace_resolution(xray.beam_params)
    factor = max(config.detector_shape)/max_axis
    shape = tuple(max(2, int(round(n/factor))) for n in config.detector_shape)
    pitch = resolution*factor
    aperture = FrontApertureConfig(aperture_method=config.aperture_method,
        aperture_shape=(len(layers), *shape), real_space_pixel_size=pitch,
        aperture_thicknesses=thicknesses, aperture_layer_names=names,
        aperture_config={
            'apertures_type': config.aperture_types, 'apertures_radius': config.aperture_radii,
            'apertures_center': config.aperture_centers, 'apertures_sigma': config.aperture_sigmas,
            'apertures_depth': config.aperture_depths, 'apertures_length': config.aperture_lengths,
            'apertures_angle': config.aperture_angles, 'apertures_ellipticity': config.aperture_ellipticities,
            'apertures_roughness': config.aperture_roughnesses,
            'apertures_roughness_modes': config.aperture_roughness_modes,
            'apertures_seed': config.aperture_seeds,
            'apertures_top_radius_factor': config.aperture_top_radius_factors,
            'aperture_taper_depth': float(thicknesses[:max(0, membrane-2)].sum()),
            'thickness_OH': float(thicknesses[:membrane].sum()),
        })
    aperture.setup()
    holes = np.average(1-aperture.return_aperture(), axis=0, weights=thicknesses)
    ny, nx = shape
    extent = ((-nx/2-.5)*pitch*1e6, (nx/2-.5)*pitch*1e6,
              (-ny/2-.5)*pitch*1e6, (ny/2-.5)*pitch*1e6)
    return holes, extent


def group_for(name):
    """Assign every backend configuration field to a thematic panel."""
    if name in {'oversampling', 'random_seed', 'use_roi',
                'magnetic_pattern_max_grid_axis', 'magnetic_pattern_use_roi'}:
        return 'Sample structure and grid'
    if name.startswith(('pattern_', 'magnetic_')):
        return 'Magnetic pattern'
    if name.startswith('aperture_'):
        return 'Holography mask'
    if name.startswith(('xray_', 'illumination_')):
        return 'X-rays and illumination'
    if name.startswith('beamstop_'):
        return 'Beamstop'
    if name in {'detector_params', 'detector_noise_rms', 'detector_quantum_efficiency'} or name.startswith(
            ('artifacts_', 'measurement_', 'save_detected')):
        return 'Acquisition and sensor response'
    if name.startswith('detector_'):
        return 'Detector geometry'
    if name.startswith(('recipe', 'sample_', 'optical_', 'refractive_')):
        return 'Sample structure and grid'
    return 'Propagation'


# Duplicate aliases and Python-only object controls do not belong in the GUI.
HIDDEN_FIELDS = {'pattern_config_length', 'pattern_type', 'detector_noise_rms',
                 'detector_quantum_efficiency', 'optical_responses'}


def parameter_is_active(path, preparation='random', order=0., disorder=0., attempts=1):
    """Return whether a preparation/calibration setting affects the selected run."""
    if path[0] in HIDDEN_FIELDS:
        return False
    if path[0] != 'pattern_config' or len(path) < 2:
        return True
    key = path[1]
    if key == 'initial_order':
        return preparation != 'random'
    if key in {'worm_length', 'worm_waviness', 'initial_domain_count'}:
        return preparation == 'worms' and order > 0
    if key == 'disorder_length':
        return disorder > 0
    if key == 'length_tolerance':
        return attempts > 1
    return True


CHOICES = {
    'propagator_method': ('Scalar', 'Jones', 'Stokes'),
    'pattern_type': ('constrained_domain_pattern',),
    'initial_state': ('random', 'stripes', 'hexagonal', 'worms'),
    'acquisition_mode': ('accumulated', 'per_frame'),
    'illumination_function': ('None', 'gaussian'),
    'beamstop_method': ('None', 'circular'),
    'detector_propagation_method': ('fraunhofer', 'rayleigh_sommerfeld'),
    'sample_tilt_axis': ('x', 'y'),
    'propagation_padding_mode': ('edge', 'reflect', 'constant'),
    'propagation_absorber_profile': ('cosine', 'smoothstep', 'quadratic', 'linear'),
}


class SimulatorApp:
    """Build a Tk interface; the simulation process never accesses its widgets."""

    def __init__(self, root):
        import tkinter as tk
        from tkinter import ttk
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
        self.tk, self.ttk, self.root = tk, ttk, root
        self.temp = tempfile.TemporaryDirectory(prefix='fomocid-gui-')
        self.process = None
        self.poll_job = None
        self.result = None
        self.result_path = None
        self.editors = {}
        self.editor_rows = {}
        self.dictionary_overrides = {}
        self.plot_settings = {}
        self.context = mp.get_context('spawn')
        root.title('Fomocid — single experiment')
        root.geometry('1450x950')
        root.protocol('WM_DELETE_WINDOW', self.close)
        toolbar = ttk.Frame(root, padding=8)
        toolbar.pack(fill='x')
        self.run_button = ttk.Button(toolbar, text='Simulate', command=self.simulate)
        self.run_button.pack(side='left')
        self.cancel_button = ttk.Button(toolbar, text='Cancel', command=self.cancel, state='disabled')
        self.cancel_button.pack(side='left', padx=5)
        self.export_button = ttk.Button(toolbar, text='Export HDF5…', command=self.export, state='disabled')
        self.export_button.pack(side='left')
        self.random_recipe = tk.BooleanVar(value=False)
        ttk.Label(toolbar, text='Lengths: metres   Energy: eV   Angles: radians   Recipe: nm').pack(side='left')
        panes = ttk.Panedwindow(root, orient='horizontal')
        panes.pack(fill='both', expand=True)
        left, right = ttk.Frame(panes), ttk.Frame(panes)
        panes.add(left, weight=1)
        panes.add(right, weight=3)
        scroll = tk.Canvas(left, width=450, highlightthickness=0)
        scrollbar = ttk.Scrollbar(left, orient='vertical', command=scroll.yview)
        scroll.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side='right', fill='y')
        scroll.pack(side='left', fill='both', expand=True)
        settings = ttk.Frame(scroll)
        window = scroll.create_window((0, 0), window=settings, anchor='nw')
        settings.bind('<Configure>', lambda _: scroll.configure(scrollregion=scroll.bbox('all')))
        scroll.bind('<Configure>', lambda e: scroll.itemconfigure(window, width=e.width))
        config = default_config()
        section_frames = {}
        for title in SECTIONS:
            frame = self.section(settings, title)
            section_frames[title] = frame
            if title == 'Sample structure and grid':
                ttk.Checkbutton(frame, text='Random recipe', variable=self.random_recipe).pack(anchor='w')
            for field in fields(config):
                if field.name in HIDDEN_FIELDS or group_for(field.name) != title:
                    continue
                value = getattr(config, field.name)
                if isinstance(value, dict) and value:
                    nested = self.section(frame, {
                        'pattern_config': 'Domain generation controls',
                        'pattern_config_length': 'Legacy length overrides',
                        'artifacts_config': 'Photon spreading and sensor artifacts',
                        'measurement_config': 'Exposure and accumulation',
                        'detector_params': 'Gain, readout and ADC',
                    }.get(field.name, field.name.replace('_', ' ')))
                    for key, entry in value.items():
                        if key != 'nonlinear_response':
                            self.add_editor(nested, (field.name, key), entry)
                    self.ttk.Label(nested, text='Additional dictionary overrides (Python literal)').pack(anchor='w')
                    overrides = self.tk.StringVar(value='{}')
                    self.ttk.Entry(nested, textvariable=overrides).pack(fill='x')
                    self.dictionary_overrides[field.name] = overrides
                else:
                    self.add_editor(frame, (field.name,), value)
        preview_panel = self.section(section_frames['Holography mask'], 'Aperture preview (depth average)')
        self.preview_figure = Figure(figsize=(4, 3), constrained_layout=True)
        self.preview_canvas = FigureCanvasTkAgg(self.preview_figure, master=preview_panel)
        self.preview_canvas.get_tk_widget().pack(fill='x')
        NavigationToolbar2Tk(self.preview_canvas, preview_panel).update()
        ttk.Button(preview_panel, text='Refresh aperture preview', command=self.refresh_preview).pack(fill='x')
        self.preview_note = tk.StringVar(value='Refresh after editing the recipe or aperture geometry.')
        ttk.Label(preview_panel, textvariable=self.preview_note, wraplength=380).pack(fill='x')
        response_panel = self.section(section_frames['Acquisition and sensor response'],
                                      'Optional nonlinear response')
        self.response = tk.StringVar(value='linear')
        ttk.Combobox(response_panel, textvariable=self.response, values=('linear', 'saturating'), state='readonly').pack(fill='x')
        ttk.Label(response_panel, text='Saturation scale (ADU; x / (1 + x/scale))').pack(anchor='w')
        self.response_scale = tk.StringVar(value='50000')
        ttk.Entry(response_panel, textvariable=self.response_scale).pack(fill='x')
        view_controls = ttk.Frame(right)
        view_controls.pack(fill='x')
        self.polarization = tk.StringVar(value='CR')
        self.exit_view = tk.StringVar(value='amplitude')
        self.log_view = tk.BooleanVar(value=True)
        for label, var, values in [('Polarization', self.polarization, ('CL', 'CR', 'CL−CR', 'CL/CR')),
                                   ('Exit wave', self.exit_view, ('amplitude', 'phase', 'real', 'imaginary'))]:
            ttk.Label(view_controls, text=label).pack(side='left', padx=5)
            control = ttk.Combobox(view_controls, textvariable=var, values=values, width=12, state='readonly')
            control.pack(side='left')
            control.bind('<<ComboboxSelected>>', lambda _: self.draw())
        ttk.Checkbutton(view_controls, text='Log hologram intensity', variable=self.log_view,
                        command=self.draw).pack(side='left', padx=8)
        color_controls = ttk.Frame(right)
        color_controls.pack(fill='x')
        self.plot_target = tk.StringVar(value='Magnetic pattern')
        self.plot_names = ('Magnetic pattern', 'Exit wave', 'Ideal hologram', 'Detected hologram')
        self.colormap = tk.StringVar(value='auto')
        self.color_scale = tk.StringVar(value='auto')
        self.color_min, self.color_max = tk.StringVar(), tk.StringVar()
        for variable, options, width in [(self.plot_target, self.plot_names, 18),
                (self.colormap, ('auto', 'gray', 'viridis', 'inferno', 'magma', 'RdBu_r', 'twilight'), 10),
                (self.color_scale, ('auto', 'linear', 'log', 'symlog'), 8)]:
            widget = ttk.Combobox(color_controls, textvariable=variable, values=options, width=width, state='readonly')
            widget.pack(side='left', padx=2)
            if variable is self.plot_target:
                widget.bind('<<ComboboxSelected>>', lambda _: self.load_plot_controls())
        for label, variable in [('Min', self.color_min), ('Max', self.color_max)]:
            ttk.Label(color_controls, text=label).pack(side='left')
            ttk.Entry(color_controls, textvariable=variable, width=9).pack(side='left')
        ttk.Button(color_controls, text='Apply colors', command=self.apply_plot_controls).pack(side='left', padx=3)
        ttk.Button(color_controls, text='Reset view', command=self.reset_view).pack(side='left')
        self.figure = Figure(figsize=(10, 7), constrained_layout=True)
        self.canvas = FigureCanvasTkAgg(self.figure, master=right)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        self.plot_toolbar = NavigationToolbar2Tk(self.canvas, right)
        self.plot_toolbar.update()
        self.canvas.mpl_connect('scroll_event', self.scroll_zoom)
        self.status = tk.StringVar(value='Ready. Set parameters and click Simulate.')
        ttk.Label(root, textvariable=self.status, wraplength=1400, padding=6).pack(fill='x')
        self.log = tk.Text(root, height=7, state='disabled', wrap='word')
        self.log.pack(fill='x')
        for key in ('initial_state', 'initial_order', 'disorder', 'max_length_iterations'):
            self.editors[('pattern_config', key)][0].trace_add('write', lambda *_: self.update_parameter_visibility())
        self.update_parameter_visibility()
        self.draw()
        self.root.after(100, self.refresh_preview)

    def section(self, parent, title):
        """Create a compact collapsible settings section."""
        outer = self.ttk.Frame(parent, padding=3)
        outer.pack(fill='x', anchor='n')
        body = self.ttk.Frame(outer, padding=(8, 3))
        expanded = self.tk.BooleanVar(value=False)
        def toggle():
            if expanded.get():
                body.pack(fill='x')
            else:
                body.pack_forget()
        self.ttk.Checkbutton(outer, text=title, variable=expanded, command=toggle).pack(anchor='w')
        return body

    def add_editor(self, parent, path, value):
        """Create an editable field for each scalar or compound backend setting."""
        row = self.ttk.Frame(parent)
        row.pack(fill='x', pady=2)
        self.ttk.Label(row, text=path[-1], width=30).pack(anchor='w')
        text = value if isinstance(value, str) else repr(value)
        variable = self.tk.StringVar(value=text)
        choices = ('True', 'False') if isinstance(value, bool) else CHOICES.get(path[-1])
        if choices:
            control = self.ttk.Combobox(row, textvariable=variable, values=choices, state='readonly')
        else:
            control = self.ttk.Entry(row, textvariable=variable)
        control.pack(fill='x')
        self.editors[path] = (variable, value)
        self.editor_rows[path] = row

    def current_parameter_conditions(self):
        """Read dependency controls while tolerating temporarily incomplete typing."""
        def value(key, fallback):
            variable, template = self.editors[('pattern_config', key)]
            try:
                return parse_value(variable.get(), template)
            except ValueError:
                return fallback
        return dict(preparation=value('initial_state', 'random'),
                    order=value('initial_order', 0.), disorder=value('disorder', 0.),
                    attempts=value('max_length_iterations', 1))

    def update_parameter_visibility(self):
        """Show only controls used by the current domain preparation."""
        conditions = self.current_parameter_conditions()
        for path, row in self.editor_rows.items():
            if not parameter_is_active(path, **conditions):
                row.pack_forget()
            elif not row.winfo_manager():
                siblings = row.master.winfo_children()
                later = siblings[siblings.index(row)+1:]
                following = next((child for child in later if child.winfo_manager() == 'pack'), None)
                options = dict(fill='x', pady=2)
                if following is not None:
                    options['before'] = following
                row.pack(**options)

    def get_config(self):
        """Validate editor literals and assemble one explicit backend configuration."""
        values = vars(default_config()).copy()
        conditions = self.current_parameter_conditions()
        for path, (variable, template) in self.editors.items():
            if not parameter_is_active(path, **conditions):
                continue
            try:
                text = variable.get()
                value = None if text == 'None' and path[-1] in CHOICES else parse_value(text, template)
            except ValueError as error:
                raise ValueError(f'{" / ".join(path)}: {error}') from error
            if len(path) == 1:
                values[path[0]] = value
            else:
                values[path[0]][path[1]] = value
        for name, variable in self.dictionary_overrides.items():
            values[name].update(parse_value(variable.get(), {}))
        if self.response.get() == 'saturating':
            scale = float(self.response_scale.get())
            if not np.isfinite(scale) or scale <= 0:
                raise ValueError('Saturation scale must be a finite positive number')
            values['detector_params']['nonlinear_response'] = SaturatingResponse(scale)
        if self.random_recipe.get():
            values['recipe'] = sample_multilayer_recipe(np.random.default_rng(values['random_seed']))
        shape = values['detector_shape']
        if len(shape) != 2 or any(not isinstance(n, int) or n < 2 for n in shape):
            raise ValueError('detector_shape must contain two integers >= 2')
        for name in ('detector_distance', 'detector_pixel_size', 'xray_energy'):
            if not np.isfinite(values[name]) or values[name] <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if values['oversampling'] < 1:
            raise ValueError('oversampling must be positive')
        return HologramPipelineConfig(**values)

    def simulate(self):
        """Start a background simulation, retaining the last successful result."""
        from tkinter import messagebox
        if self.process is not None:
            return
        try:
            config = self.get_config()
        except Exception as error:
            messagebox.showerror('Invalid parameters', str(error))
            return
        self.messages = self.context.Queue()
        self.pending = Path(self.temp.name) / 'pending.h5'
        self.log_path = Path(self.temp.name) / 'simulation.log'
        self.log_path.write_text('')
        self.log_offset = 0
        self.process = self.context.Process(target=worker,
            args=(config, self.pending, self.log_path, self.messages))
        self.process.start()
        self.run_button.configure(state='disabled')
        self.cancel_button.configure(state='normal')
        self.status.set('Simulating… Previous plots remain visible until the new result is ready.')
        self.poll_job = self.root.after(150, self.poll)

    def poll(self):
        """Refresh progress without blocking Tk's event loop."""
        from queue import Empty
        from tkinter import messagebox
        self.poll_job = None
        if self.process is None:
            return
        if self.log_path.exists():
            with self.log_path.open() as stream:
                stream.seek(self.log_offset)
                text = stream.read()
                self.log_offset = stream.tell()
            if text:
                self.log.configure(state='normal')
                self.log.insert('end', text)
                self.log.see('end')
                self.log.configure(state='disabled')
        try:
            kind, payload = self.messages.get_nowait()
        except Empty:
            if not self.process.is_alive():
                # Queue feeder can finish just after process exit.
                try:
                    kind, payload = self.messages.get(timeout=.1)
                except Empty:
                    kind, payload = 'error', f'Simulation process exited with code {self.process.exitcode}.'
            else:
                self.poll_job = self.root.after(150, self.poll)
                return
        self.process.join(timeout=.2)
        self.process = None
        self.run_button.configure(state='normal')
        self.cancel_button.configure(state='disabled')
        if kind == 'done':
            try:
                result = read_result(payload)
                destination = Path(self.temp.name) / 'last_successful.h5'
                Path(payload).replace(destination)
                self.result, self.result_path = result, destination
                self.export_button.configure(state='normal')
                self.draw()
                c = result['classification']
                self.status.set(f"{c['state']} — B={c['bubble_count']}, S={c['stripe_count']}; "
                    f"pixel pitch={result['pitch']*1e9:.2f} nm; "
                    f"nematic={c.get('stripe_nematic_order')}, sixfold={c.get('local_sixfold_order')}. "
                    f"Recipe: {result['recipe']}")
            except Exception:
                self.status.set('Cannot read the new result. Previous successful result is retained.')
                messagebox.showerror('Cannot read result', traceback.format_exc())
        else:
            self.status.set('Simulation failed. Previous successful result is retained.')
            messagebox.showerror('Simulation failed', payload)

    def cancel(self):
        """Stop only the isolated simulation, preserving the last complete result."""
        if self.poll_job is not None:
            self.root.after_cancel(self.poll_job)
            self.poll_job = None
        if self.process is not None:
            self.process.terminate()
            self.process.join()
            self.process = None
            self.run_button.configure(state='normal')
            self.cancel_button.configure(state='disabled')
            self.status.set('Cancelled. Previous successful result is retained.')

    def refresh_preview(self):
        """Render a physical, thickness-weighted aperture preview without relaxation."""
        try:
            config = self.get_config()
            image, extent = aperture_preview(config)
            self.preview_figure.clear()
            axis = self.preview_figure.subplots()
            artist = axis.imshow(image, origin='lower', extent=extent, cmap='gray', vmin=0, vmax=1)
            for kind, (y, x) in zip(config.aperture_types, config.aperture_centers):
                axis.annotate(kind, (x*1e6, y*1e6), color='tab:red', fontsize=8)
            axis.set(xlabel='x (µm)', ylabel='y (µm)', title='Depth-averaged open fraction')
            self.preview_figure.colorbar(artist, ax=axis, shrink=.7)
            self.preview_canvas.draw_idle()
            self.preview_note.set('0 = material throughout depth; 1 = open throughout depth. '
                                  'Preview sampling is coarser than simulation.')
        except Exception as error:
            self.preview_note.set(f'Preview unavailable: {error}')

    def load_plot_controls(self):
        """Load independent display settings for the selected image."""
        index = self.plot_names.index(self.plot_target.get())
        settings = self.plot_settings.get(index, {})
        self.colormap.set(settings.get('cmap', 'auto'))
        self.color_scale.set(settings.get('scale', 'auto'))
        self.color_min.set(settings.get('vmin', ''))
        self.color_max.set(settings.get('vmax', ''))

    def apply_plot_controls(self):
        """Validate and apply colormap, scale and optional manual intensity limits."""
        from tkinter import messagebox
        try:
            settings = dict(cmap=self.colormap.get(), scale=self.color_scale.get())
            for key, variable in [('vmin', self.color_min), ('vmax', self.color_max)]:
                if variable.get().strip():
                    value = float(variable.get())
                    if not np.isfinite(value):
                        raise ValueError('Color limits must be finite')
                    if settings['scale'] == 'log' and value <= 0:
                        raise ValueError('Logarithmic color limits must be positive')
                    settings[key] = value
            if 'vmin' in settings and 'vmax' in settings and settings['vmin'] >= settings['vmax']:
                raise ValueError('Minimum color limit must be smaller than maximum')
            self.plot_settings[self.plot_names.index(self.plot_target.get())] = settings
            self.draw()
        except Exception as error:
            messagebox.showerror('Invalid color scale', str(error))

    def reset_view(self):
        """Restore full image extents without discarding color settings."""
        self._view_result_id = None
        self.draw()

    def scroll_zoom(self, event):
        """Zoom the image under the pointer with the mouse wheel."""
        if event.inaxes not in getattr(self, 'image_axes', ()) or event.xdata is None or event.ydata is None:
            return
        axis = event.inaxes
        factor = 1/1.25 if event.button == 'up' else 1.25
        x, y = event.xdata, event.ydata
        left, right = axis.get_xlim()
        bottom, top = axis.get_ylim()
        axis.set_xlim(x+(left-x)*factor, x+(right-x)*factor)
        axis.set_ylim(y+(bottom-y)*factor, y+(top-y)*factor)
        self.canvas.draw_idle()

    def draw(self):
        """Plot magnetic field, complex exit wave and both hologram sources."""
        from matplotlib.colors import LogNorm, SymLogNorm
        keep_view = getattr(self, '_view_result_id', None) == id(self.result)
        limits = [(axis.get_xlim(), axis.get_ylim()) for axis in getattr(self, 'image_axes', ())] if keep_view else []
        self.figure.clear()
        axes = self.figure.subplots(2, 2).ravel()
        self.image_axes = axes
        self._view_result_id = id(self.result)
        titles = ['Magnetic pattern: smoothed mz × OH', 'Exit wave', 'Ideal hologram', 'Detected hologram']
        if self.result is None:
            for axis, title in zip(axes, titles):
                axis.set_title(title)
                axis.set_axis_off()
            self.canvas.draw_idle()
            return
        pol = self.polarization.get()
        def field(source):
            return polarization_view(self.result['CR'][source], self.result['CL'][source], pol)
        wave = field('exit_wave')
        mode = self.exit_view.get()
        wave = {'amplitude': np.abs, 'phase': np.angle,
                'real': np.real, 'imaginary': np.imag}[mode](wave)
        arrays = [self.result['magnetic'], wave, field('ideal'), field('detected')]
        for i, (axis, array, title) in enumerate(zip(axes, arrays, titles)):
            kwargs = dict(origin='lower', interpolation='nearest', cmap='inferno')
            if i == 0:
                kwargs.update(cmap='RdBu_r', vmin=-1, vmax=1)
            elif i == 1 and mode == 'phase':
                if pol == 'CL/CR':
                    # Magnetic phase differences are often far below one radian.
                    # A full +/-pi scale makes a valid complex ratio look flat.
                    finite_phase = np.abs(array[np.isfinite(array)])
                    limit = max(float(finite_phase.max()) if finite_phase.size else 0., 1e-9)
                    kwargs.update(cmap='RdBu_r', vmin=-limit, vmax=limit)
                else:
                    kwargs.update(cmap='twilight', vmin=-np.pi, vmax=np.pi)
            elif i >= 2 and self.log_view.get():
                finite = np.abs(array[np.isfinite(array)])
                maximum = finite.max() if finite.size else 0.
                if maximum > 0:
                    if np.any(array < 0):
                        kwargs.update(norm=SymLogNorm(maximum*1e-4, vmin=-maximum, vmax=maximum), cmap='RdBu_r')
                    else:
                        kwargs.update(norm=LogNorm(vmin=max(maximum*1e-6, np.finfo(float).tiny), vmax=maximum))
            settings = getattr(self, 'plot_settings', {}).get(i, {})
            if settings.get('cmap', 'auto') != 'auto':
                kwargs['cmap'] = settings['cmap']
            scale = settings.get('scale', 'auto')
            manual_limits = any(key in settings for key in ('vmin', 'vmax'))
            if scale != 'auto' or manual_limits:
                current_norm = kwargs.pop('norm', None)
                lower = kwargs.pop('vmin', current_norm.vmin if current_norm is not None else float(np.nanmin(array)))
                upper = kwargs.pop('vmax', current_norm.vmax if current_norm is not None else float(np.nanmax(array)))
                lower, upper = settings.get('vmin', lower), settings.get('vmax', upper)
                if upper <= lower:
                    upper = lower + max(abs(lower)*1e-6, 1e-12)
                use_log = scale == 'log' or (scale == 'auto' and isinstance(current_norm, LogNorm))
                use_signed = scale == 'symlog' or (scale == 'auto' and isinstance(current_norm, SymLogNorm))
                if use_log:
                    positives = array[array > 0]
                    if positives.size:
                        lower = max(lower, float(positives.min()), np.finfo(float).tiny)
                        upper = max(upper, lower*(1+1e-6))
                        kwargs['norm'] = LogNorm(lower, upper)
                elif use_signed:
                    kwargs['norm'] = SymLogNorm(max(abs(lower), abs(upper), 1e-12)*1e-4, vmin=lower, vmax=upper)
                else:
                    kwargs.update(vmin=lower, vmax=upper)
            image = axis.imshow(array, **kwargs)
            axis.set_title(title + (f' ({mode}, {pol})' if i == 1 else f' ({pol})' if i >= 2 else ''))
            axis.set_xlabel('x pixel')
            axis.set_ylabel('y pixel')
            colorbar = self.figure.colorbar(image, ax=axis, shrink=.75)
            if i == 1:
                colorbar.set_label('Phase (rad)' if mode == 'phase' else
                    'Amplitude ratio' if mode == 'amplitude' and pol == 'CL/CR' else mode.capitalize())
            if i < len(limits):
                axis.set_xlim(limits[i][0])
                axis.set_ylim(limits[i][1])
        if hasattr(self, 'plot_toolbar'):
            self.plot_toolbar.update()
        self.canvas.draw_idle()

    def export(self):
        """Copy the last complete experiment to a user-selected HDF5 file."""
        from tkinter import filedialog, messagebox
        if self.result_path is None:
            return
        destination = filedialog.asksaveasfilename(defaultextension='.h5',
            filetypes=[('HDF5', '*.h5')], initialfile='single_experiment.h5')
        if destination:
            try:
                shutil.copy2(self.result_path, destination)
                self.status.set(f'Exported complete experiment to {destination}')
            except OSError as error:
                messagebox.showerror('Export failed', str(error))

    def close(self):
        """Stop the child process before removing temporary results and closing."""
        from tkinter import messagebox
        if self.process is not None and not messagebox.askyesno('Close simulator?', 'Cancel the running simulation and close?'):
            return
        self.cancel()
        self.temp.cleanup()
        self.root.destroy()


def main():
    """Launch the desktop app without starting a simulation automatically."""
    import tkinter as tk
    try:
        root = tk.Tk()
    except tk.TclError as error:
        raise SystemExit('Cannot open the desktop display. Run this GUI in a graphical '
                         'desktop session with Tk support. Details: ' + str(error)) from error
    SimulatorApp(root)
    root.mainloop()


if __name__ == '__main__':
    mp.freeze_support()
    main()

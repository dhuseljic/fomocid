"""Branch magnetic textures into exit waves, geometries and detector acquisitions."""
from copy import copy, deepcopy
from dataclasses import replace
import json
import time

import numpy as np

from .hologram_pipeline import HologramPipeline
from ..simulation_configuration import XRayConfig


class HierarchicalHologramPipeline(HologramPipeline):
    """Reuse costly physics across a Cartesian tree of explicit variants.

    ``n_samples`` counts independent magnetic-pattern parents. Exit variants
    use config field names and may change energy or reference-hole design.
    Geometry variants and acquisition variants use DetectorConfig field names.
    Nested acquisition dictionaries merge with the sampled parent settings.
    The object-hole geometry and the sample grid must remain fixed across exit
    variants. All distances are metres, energies eV and exposures seconds.
    """

    EXIT_KEYS = {'xray_energy', 'aperture_radii', 'aperture_centers',
                 'aperture_sigmas', 'aperture_lengths', 'aperture_angles',
                 'aperture_ellipticities', 'aperture_top_radius_factors'}
    GEOMETRY_KEYS = {'sample_to_detector_distance', 'pixel_size', 'detector_center',
                     'use_detector_pixel_footprint', 'detector_pixel_footprint_samples',
                     'ignore_flat_detector_curvature', 'detector_propagation_method'}
    ACQUISITION_KEYS = {'measurement_config', 'artifacts_config', 'detector_params'}

    def __init__(self, *args, exit_variants=None, geometry_variants=None,
                 acquisition_variants=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.exit_variants = self._variants(exit_variants, self.EXIT_KEYS)
        self.geometry_variants = self._variants(geometry_variants, self.GEOMETRY_KEYS)
        self.acquisition_variants = self._variants(acquisition_variants, self.ACQUISITION_KEYS)
        self._leaf_index = 0

    @staticmethod
    def _variants(values, allowed):
        """Validate explicit variant dictionaries before any computation."""
        values = deepcopy([{}] if values is None else list(values))
        if not values:
            raise ValueError('Variant lists must not be empty')
        for value in values:
            unknown = set(value) - allowed
            if unknown:
                raise ValueError(f'Parameters belong to a different stage: {sorted(unknown)}')
        return values

    def _write_pipeline_config(self, h5):
        super()._write_pipeline_config(h5)
        h5.attrs['pipeline_type'] = 'hierarchical_hologram'
        h5.attrs['hierarchy_variants_json'] = json.dumps(dict(
            exit=self.exit_variants, geometry=self.geometry_variants,
            acquisition=self.acquisition_variants), default=str)
        self._leaf_index = 0
        h5.attrs['magnetic_parent_count'] = self.n_samples
        h5.attrs['acquisition_pair_count'] = (self.n_samples * len(self.exit_variants)
            * len(self.geometry_variants) * len(self.acquisition_variants))

    def _sample_params(self):
        return deepcopy(self._parent_params)

    def _simulate_one(self, h5, idx):
        original = self.config
        self._pattern = None
        self._resolution = None
        # Sample independent parents once; downstream variants are explicit.
        parent_params = super()._sample_params()
        self._base_energy = parent_params['energy']
        try:
            for exit_index, variant in enumerate(self.exit_variants):
                self.config = replace(original, **variant)
                self._parent_params = deepcopy(parent_params)
                for key, value in variant.items():
                    if key != 'xray_energy':
                        old = parent_params['aperture_config'][key]
                        types = parent_params['aperture_config']['aperture_types']
                        if len(value) != len(old) or any(
                                not np.array_equal(value[i], old[i])
                                for i, kind in enumerate(types) if kind == 'OH'):
                            raise ValueError('Exit variants must preserve object-hole geometry')
                    if key == 'xray_energy':
                        self._parent_params['energy'] = value
                    else:
                        self._parent_params['aperture_config'][key] = deepcopy(value)
                self._ids = (idx, exit_index)
                self._propagators = {}
                self._ideals = {}
                super()._simulate_one(h5, idx)
        finally:
            self.config = original
            self._pattern = None
            self._propagators = {}
            self._ideals = {}

    def _sampling_resolution(self, detector_config, xray_config):
        # Resolve the grid at the parent's energy, never the branch energy.
        if self._resolution is None:
            reference = XRayConfig(energy=self._base_energy,
                                  pol='CR', photon_flux=1.)
            reference.setup()
            self._resolution = super()._sampling_resolution(detector_config, reference)
        detector_config.calc_realspace_resolution(xray_config.beam_params)
        return self._resolution

    def _generate_pattern(self, pattern_config, callback):
        signature = (tuple(pattern_config.shape), pattern_config.real_space_pixel_size,
                     repr(pattern_config.pattern_config))
        if self._pattern is None:
            self._pattern_signature = signature
            self._pattern = super()._generate_pattern(pattern_config, callback)
        elif signature != self._pattern_signature:
            raise ValueError('Exit variants must retain the same OH and magnetic generation grid')
        return self._pattern

    def _setup_propagator(self, propagator_config, pol):
        if pol not in self._propagators:
            self._propagators[pol] = super()._setup_propagator(propagator_config, pol)
        return self._propagators[pol]

    def _detect_ideal(self, detector_config, propagator_config, pol):
        if pol not in self._ideals:
            super()._detect_ideal(detector_config, propagator_config, pol)
            self._ideals[pol] = copy(detector_config.hologram_exp)
        else:
            detector_config.propagator = propagator_config
            detector_config.wavefront = propagator_config.return_wavefront()
            detector_config.hologram_exp = copy(self._ideals[pol])
        model = detector_config.hologram_exp
        # Reset acquisition settings through the validated canonical helpers.
        artifacts, params = model._normalize_counts_per_photon(
            detector_config.artifacts_config, detector_config.detector_params)
        model._apply_measurement_config(detector_config.measurement_config)
        model._apply_artifacts_config(artifacts)
        model._apply_detector_params(model._normalize_detector_param_aliases(params))

    def _process_prepared(self, h5, idx, p, metadata, stage_times, t_stage,
                          sample_config, illumination_config, detector_config,
                          xray_config, sample_shape, real_space_pixel_size,
                          supportmask, magnetic_pattern_oh, front_aperture_config):
        base = detector_config
        parent, exit_index = self._ids
        for geometry_index, geometry in enumerate(self.geometry_variants):
            self._ideals = {}
            geometry_detector = replace(base, **geometry)
            geometry_detector.setup()
            geometry_detector.calc_realspace_resolution(xray_config.beam_params)
            geometry_supportmask = front_aperture_config.create_supportmask(
                output_shape=geometry_detector.shape,
                output_pixel_size=geometry_detector.detector_layout.real_space_resolution)
            for acquisition_index, acquisition in enumerate(self.acquisition_variants):
                settings = {name: {**getattr(base, name), **acquisition.get(name, {})}
                            for name in self.ACQUISITION_KEYS}
                # Each leaf gets fresh noise; CR/CL use adjacent seeds.
                settings['detector_params']['noise_seed'] = int(np.random.randint(0, 2**31-2))
                detector = copy(geometry_detector)
                for name, value in settings.items():
                    setattr(detector, name, value)
                leaf_metadata = dict(metadata)
                leaf_metadata.update(self._detector_metadata(detector))
                for name, value in dict(pattern_id=parent,
                        exit_wave_id=f'{parent}:{exit_index}',
                        ideal_hologram_id=f'{parent}:{exit_index}:{geometry_index}',
                        acquisition_index=acquisition_index).items():
                    leaf_metadata[f'hierarchy/{name}'] = value
                leaf_metadata['hierarchy/exit_variant_json'] = json.dumps(self.exit_variants[exit_index])
                leaf_metadata['hierarchy/geometry_variant_json'] = json.dumps(geometry)
                leaf_metadata['hierarchy/acquisition_variant_json'] = json.dumps(acquisition, default=str)
                leaf_metadata['hierarchy/reused_exit_wave'] = geometry_index > 0 or acquisition_index > 0
                leaf_metadata['hierarchy/reused_ideal_hologram'] = acquisition_index > 0
                if self.verbose and self.detailed_logging:
                    print(f'    leaf {self._leaf_index:05d}: pattern={parent}, exit={exit_index}, '
                          f'geometry={geometry_index}, acquisition={acquisition_index}', flush=True)
                super()._process_prepared(h5, self._leaf_index, p, leaf_metadata,
                    list(stage_times) if geometry_index == acquisition_index == 0 else [],
                    time.perf_counter(), sample_config, illumination_config, detector,
                    xray_config, sample_shape, real_space_pixel_size, geometry_supportmask, magnetic_pattern_oh, front_aperture_config)
                self._leaf_index += 1

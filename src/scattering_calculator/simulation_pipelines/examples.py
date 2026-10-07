"""Reference parameter sets, all returned in the standard ExperimentConfig format.

Presets supply values only. They do not define a second API or execution path.
"""
import numpy as np
from . import simulation_configuration as sim
from .experiment import ExperimentConfig

def fth_experiment():
    """Return the fth reference using the standard component sections.

    Parameters
    ----------
    None
        Uses the stored experiment state.

    Returns
    -------
    ExperimentConfig
        Standard component sections populated with reference values.
    """
    experiment = ExperimentConfig(
        xray=sim.XRayConfig(energy=778., photon_flux=1., pol="CR",
                           coherence_length=(100e-6, 100e-6)),
        simulation=sim.SimulationConfig(shape=(192,192), real_space_pixel_size=10e-9),
        sample=sim.SampleConfig(recipe="Au(300)/SiN(20)/Pt(2)/Co(10)/Pt(2)",
                                max_slice_thickness=5e-9, sample_name="Normal-incidence FTH"),
        magnetic_pattern=sim.MagneticPatternConfig(
            pattern_type_method="smooth_domains", pattern_config={"period":110e-9}),
        aperture=sim.FrontApertureConfig(aperture_method="FTH_circular", aperture_config={
            "apertures_type":["OH","RH"], "apertures_radius":[180e-9,30e-9],
            "apertures_center":[(0.,0.),(0.,600e-9)], "apertures_sigma":[0.,0.],
            "apertures_top_radius_factor":[1.,1.], "thickness_OH":300e-9,
        }),
        illumination=sim.IlluminationConfig(illumination_function="gaussian", illumination_config={
            "center":(0.,0.), "distance":0., "fwhm":2*np.sqrt(np.log(2))*650e-9,
            "alpha_beam":(0.,0.),
        }),
        propagation=sim.SamplePropagatorConfig(propagator_method="Jones", propagator_config={
            "propagate":True, "dielectric_tensor_compact":True,
        }),
        detector=sim.DetectorConfig(shape=(256,256), pixel_size=13.5e-6,
            sample_to_detector_distance=.05, detector_center=(128,128),
            detector_propagation_method="fraunhofer", use_detector_pixel_footprint=False,
            detector_pixel_footprint_samples=3, analyzer_angle=None,
            measurement_config={"exposure_time":1.,"number_frames":1,"max_counts_per_image":None},
            detector_params={"counts_per_photon":1.,"quantum_efficiency":1.,
                "readout_noise_average":0.,"readout_noise_sigma":2.,"detector_threshold":16000.,
                "noise_seed":17},
            artifacts_config={"sigma_photon":0.,"camera_seed":31,"average_hot_pixels":12.,
                              "average_cold_pixels":12.,"cosmic_rays_per_second":2.}),
        beamstop=sim.BeamstopConfig(bs_method="circular",bs_detector_distance=.01,
                                  bs_config={"radius":40.5e-6}),
        energies_eV=tuple(np.arange(772.,801.)),
    )
    return experiment

def magnon_experiment():
    """Return the magnon reference using the standard component sections.

    Parameters
    ----------
    None
        Uses the stored experiment state.

    Returns
    -------
    ExperimentConfig
        Standard component sections populated with reference values.
    """
    experiment = ExperimentConfig(
        xray=sim.XRayConfig(energy=778., photon_flux=1., pol="LH",
                           coherence_length=(1.,1.), linear_polarization_angle=0.),
        simulation=sim.SimulationConfig(shape=(384,384), real_space_pixel_size=2e-9),
        sample=sim.SampleConfig(recipe="Co(30)",max_slice_thickness=2e-9,sample_name="Co magnon-like film"),
        magnetic_pattern=sim.MagneticPatternConfig(pattern_type_method="magnon_wave",pattern_config={
            "period":40e-9, 'amplitude':.1, 'angle':0.,
            "inplane_angle":0., "phase":0.,
        }),
        aperture=sim.FrontApertureConfig(aperture_method=None),
        illumination=sim.IlluminationConfig(illumination_function="gaussian",illumination_config={
            "center":(0.,0.),"distance":0.,"fwhm":2*np.sqrt(np.log(2))*90e-9,
            "alpha_beam":(0.,0.),
        }),
        propagation=sim.SamplePropagatorConfig(propagator_method="Jones",propagator_config={
            "propagate":True,"calculate_farfield":False,"farfield_oversampling":4,
        }),
        detector=sim.DetectorConfig(shape=(256,256),pixel_size=13.5e-6,
            sample_to_detector_distance=.03,detector_center=(128,128),
            detector_propagation_method="fraunhofer",analyzer_angle=np.pi/4,
            use_detector_pixel_footprint=True,detector_pixel_footprint_samples=3,
            detector_params={"counts_per_photon":1.,"quantum_efficiency":1.,
                "readout_noise_average":0.,"readout_noise_sigma":0.,"detector_threshold":1e12},
            artifacts_config={"sigma_photon":0.},
            measurement_config={"exposure_time":1.,"number_frames":1,"max_counts_per_image":None}),
        beamstop=sim.BeamstopConfig(bs_method=None),
        energies_eV=(778.,),
    )
    return experiment

def skyrmion_experiment():
    """Return the skyrmion reference using the standard component sections.

    Parameters
    ----------
    None
        Uses the stored experiment state.

    Returns
    -------
    ExperimentConfig
        Standard component sections populated with reference values.
    """
    experiment = ExperimentConfig(
        xray=sim.XRayConfig(energy=778.,photon_flux=1.,pol="CR",coherence_length=(1.,1.)),
        simulation=sim.SimulationConfig(shape=(192,192),real_space_pixel_size=.75e-9),
        sample=sim.SampleConfig(recipe="Co(269.4766)",max_slice_thickness=2e-9,
                                sample_name="Neel skyrmion tubes"),
        magnetic_pattern=sim.MagneticPatternConfig(pattern_type_method="neel_lattice",pattern_config={
            'period':12e-9,"radius":3.4e-9,
        }),
        aperture=sim.FrontApertureConfig(aperture_method=None),
        illumination=sim.IlluminationConfig(illumination_function="gaussian",illumination_config={
            "center":(0.,0.),"distance":0.,"fwhm":2*np.sqrt(np.log(2))*24e-9,"alpha_beam":(0.,0.),
        }),
        propagation=sim.SamplePropagatorConfig(propagator_method="Scalar",propagator_config={"propagate":True}),
        detector=sim.DetectorConfig(shape=(193,193),pixel_size=13.5e-6,
            sample_to_detector_distance=.007,detector_center=(96,96),
            detector_propagation_method="fraunhofer",artifacts_config={"sigma_photon":0.}),
        beamstop=sim.BeamstopConfig(bs_method=None),
        energies_eV=(778.,),
        analysis={
            "angles":np.deg2rad([-8.7947589,-4.3973795,0.,4.3973795,8.7947589]).tolist(),
            "scan_angles":np.deg2rad(np.linspace(-12,12,49)).tolist(),
            "volume_angles":np.deg2rad(np.linspace(-12,12,25)).tolist(),
            "born_n":256,"born_pixel_size":.75e-9,"volume_n":128,
            "q_bins":101,"q_limit":.85e9,"fft_nz":512,"fft_pixel_size_z":2e-9,
            "contrast_channel":"xmcd","include_cobalt":True,
        },
    )
    return experiment


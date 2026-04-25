import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.ndimage import map_coordinates
from scattering_calculator.experimental_conditions.detector import detector_layout
from scattering_calculator.utils import image_transformator
from scipy.interpolate import griddata
from scipy import signal
from scipy.ndimage import gaussian_filter




def make_tile_class_map(shape, tile_size=256, n_classes=32, seed=None):
    """
    Assign each detector tile to one of n_classes response classes.

    shape : (Ny, Nx)
    tile_size : int
    n_classes : int

    Returns
    -------
    class_map : (Ny, Nx) int
    """
    rng = np.random.default_rng(seed)

    Ny, Nx = shape
    nty = int(np.ceil(Ny / tile_size))
    ntx = int(np.ceil(Nx / tile_size))

    tile_classes = rng.integers(0, n_classes, size=(nty, ntx))

    class_map = np.zeros((Ny, Nx), dtype=np.int32)

    for iy in range(nty):
        for ix in range(ntx):
            y0 = iy * tile_size
            y1 = min((iy + 1) * tile_size, Ny)
            x0 = ix * tile_size
            x1 = min((ix + 1) * tile_size, Nx)
            class_map[y0:y1, x0:x1] = tile_classes[iy, ix]

    return class_map


def make_random_photon_kernel(
    size=11,
    sigma_range=(0.7, 1.7),
    ellipticity_range=(0.6, 1.6),
    irregularity=0.20,
    seed=None,
):
    """
    Create one irregular single-photon response kernel.
    """
    rng = np.random.default_rng(seed)

    y, x = np.indices((size, size), dtype=float)
    cy = cx = (size - 1) / 2

    x -= cx
    y -= cy

    sigma_x = rng.uniform(*sigma_range)
    sigma_y = sigma_x * rng.uniform(*ellipticity_range)
    theta = rng.uniform(0, 2 * np.pi)

    xr = np.cos(theta) * x + np.sin(theta) * y
    yr = -np.sin(theta) * x + np.cos(theta) * y

    kernel = np.exp(
        -0.5 * ((xr / sigma_x) ** 2 + (yr / sigma_y) ** 2)
    )

    # Add smooth irregularity
    noise = 1 + irregularity * rng.normal(size=(size, size))
    noise = gaussian_filter(noise, sigma=1.0)
    kernel *= noise

    kernel[kernel < 0] = 0

    # Conserve photon number
    if kernel.max() > 0:
        kernel /= kernel.max()

    return kernel.astype(np.float64)


def make_photon_kernel_bank(
    n_classes=32,
    n_variants=4,
    size=11,
    sigma_range=(0.7, 1.7),
    ellipticity_range=(0.6, 1.6),
    irregularity=0.20,
    seed=None,
):
    """
    Kernel bank with detector classes and event variants.

    Returns
    -------
    kernels : (n_classes, n_variants, size, size)
    """
    rng = np.random.default_rng(seed)

    kernels = np.empty((n_classes, n_variants, size, size), dtype=np.float64)

    for c in range(n_classes):
        for v in range(n_variants):
            kernels[c, v] = make_random_photon_kernel(
                size=size,
                sigma_range=sigma_range,
                ellipticity_range=ellipticity_range,
                irregularity=irregularity,
                seed=rng.integers(0, 2**32 - 1),
            )

    return kernels


def split_counts_into_variants(counts, n_variants, rng):
    """
    Split an integer photon-count image into n_variants images,
    preserving the total photon number per pixel.
    """
    counts = counts.astype(np.int64, copy=True)

    variants = []
    remaining = counts.copy()

    for v in range(n_variants - 1):
        p = 1.0 / (n_variants - v)
        take = rng.binomial(remaining, p)
        variants.append(take.astype(np.float64))
        remaining -= take

    variants.append(remaining.astype(np.float64))

    return variants


def photon_splat_with_spatial_classes(
    photon_counts,
    kernel_bank,
    class_map,
    rng=None,
):
    """
    Apply spatially varying single-photon response using detector classes
    and random event variants.

    Parameters
    ----------
    photon_counts : (Ny, Nx)
        Integer photon counts per detector pixel.
    kernel_bank : (n_classes, n_variants, ky, kx)
        Photon response kernels.
    class_map : (Ny, Nx)
        Detector response class per pixel.
    rng : np.random.Generator or None

    Returns
    -------
    splatted : (Ny, Nx)
        Photon image after detector charge spreading.
    """
    if rng is None:
        rng = np.random.default_rng()

    photon_counts = np.asarray(photon_counts, dtype=np.int64)
    class_map = np.asarray(class_map, dtype=np.int32)
    kernel_bank = np.asarray(kernel_bank, dtype=np.float64)

    n_classes, n_variants = kernel_bank.shape[:2]

    splatted = np.zeros_like(photon_counts, dtype=np.float64)

    for c in range(n_classes):
        mask = class_map == c

        if not np.any(mask):
            continue

        class_counts = np.zeros_like(photon_counts, dtype=np.int64)
        class_counts[mask] = photon_counts[mask]

        if class_counts.sum() == 0:
            continue

        variant_counts = split_counts_into_variants(
            class_counts,
            n_variants=n_variants,
            rng=rng,
        )

        for v, counts_v in enumerate(variant_counts):
            if counts_v.max() == 0:
                continue

            kernel = kernel_bank[c, v]
            kernel = kernel / kernel.max()

            splatted += signal.fftconvolve(counts_v, kernel, mode="same")

    return splatted




class detector_hologram:
    def __init__(self,
                 detector_layout,
                 hologram,
                 beam_parameters,
                 real_space_pixel_size,
                 beamstop,
                 ):
        self.detector_layout = detector_layout
        self.hologram = hologram
        self.beam_parameters = beam_parameters
        self.real_space_pixel_size=real_space_pixel_size
        self.beamstop=beamstop

        self.extent_real = self.detector_layout.get_detector_extent_real_space()
        self.sample_shape = self.hologram.shape
        self.q_max_sim=np.pi / self.real_space_pixel_size

        self.readout_noise_average=50
        self.readout_noise_sigma=3
        self.sigma_h_px=0.3
        self.max_counts_per_image=60e3
        self.counts_per_photon=100
        self.number_frames=50
        self.detector_threshold=64e3
        self.sigma_photon=0.75
        self.photon_n_classes = 16
        self.photon_n_variants = 6
        self.photon_tile_size = self.hologram.shape[0]
        self.photon_kernel_size = 9
        self.photon_irregularity = 2.0
        self.regenerate_photon_kernels = True



    def gnomonic_projection(self) -> NDArray[np.float64]:
        '''Apply gnomonic projection to the hologram to correct for curvature of the Ewald sphere.
        Returns
        -------
            hologram_gnomonic : ndarray of shape (Ny, Nx)
            Gnomonic-projected hologram.        
        '''
        l= self.beam_parameters.wavelength
        k = 2 * np.pi / self.beam_parameters.wavelength
        detqx = self.detector_layout.detqx
        detqy = self.detector_layout.detqy
        z = self.detector_layout.distance_sample_detector
        detx = self.detector_layout.detx
        dety = self.detector_layout.dety
   
        # generate the qx, qy coordinates in the far field based on the real-space coordinates of the illumination plane
        # these are the q of the far field before the gnomonic projection, which are given by qx = (2 * pi / real_space_pixel_size) * (nx / sample_shape[1]) and qy = (2 * pi / real_space_pixel_size) * (ny / sample_shape[0])
        #qx = (np.arange(self.sample_shape[1])) * (np.pi/self.real_space_pixel_size) 
        #qy = (np.arange(self.sample_shape[0])) * (np.pi/self.real_space_pixel_size) 
        #QX, QY = np.meshgrid(qx, qy)
        # how much is a pixel in q space
        Dq=np.pi/self.real_space_pixel_size

        # we can use detx and dety to calculate the qx, qy coordinates in the far field corresponding to the real-space coordinates of the detector pixels
        # and then we can use these to decide where to sample hologram to have a gnomonic projection effect.
        # so, self.qxy are the q-coordinates the detector is actually mapping
        # while QX,QY are the perfect q coordinates of the sample simulation.
        # to get an actual hologram, we must map hologram(QX,QY) in the (detqx, detqy) coordinates
        #we do this using interpolation

        ## we just need to rescale detqx so they are expressed in absolute pixel value
        self.hologram_detector =map_coordinates(self.hologram, 
                                                [detqy/Dq*self.hologram.shape[0]+1*self.hologram.shape[0]/2,
                                                 detqx/Dq*self.hologram.shape[1]+1*self.hologram.shape[1]/2], 
                                                 order=5, mode='constant', cval=0)
        # now we calculate the spatial coordinates on the detector plane corresponding to these qx, qy coordinates, 
   

    def add_noise(self):
        '''
        Given the hologram, the function simulates the holograms introducing drift,
         coherence effects and Poisson noise
        INPUT:
                readout_noise_average, readout_noise_sigma: readout noise of the camera
                sigma_h_px: sigma of drift in pixles. Takes into account also the spatial incoherence, so that has to be be taken ito accout too
                max_counts_per_image: max number of counts the camera can take in one image
                counts_per_photon:
                number_of_frames: number of acquired frames. The more, the lower the noise
                
        ----------
        Author: RB_2020
        '''
        rng = np.random.default_rng()
        holo=self.hologram_detector
        npx,npy=holo.shape

        # 4. simulate sample drift / vibrations AND SPATIAL INCOHERENCE
        if self.sigma_h_px > 0:
            kernel = np.outer(signal.windows.gaussian(npx, self.sigma_h_px), signal.windows.gaussian(npx, self.sigma_h_px))
            if kernel.sum() > 0:
                kernel /= kernel.sum()
                holo = signal.fftconvolve(holo,kernel,mode='same')

        # 6. adjust the maximum count to 64000 for a single image
        factor_holo=(self.max_counts_per_image)/np.amax((1.-self.beamstop.beamstop)*holo)
        holo*=factor_holo

        # consider you will have more than one frame
        holo *= self.number_frames        

        # 7.a convert detector counts to expected photon number
        expected_photons = holo / self.counts_per_photon
        expected_photons[expected_photons < 0] = 0

        # 7.b Poisson photon sampling
        photon_counts = rng.poisson(expected_photons).astype(np.int64)
 

        # 8.b photon splatting with spatial kernel classes + event variants
        if self.sigma_photon > 0:
            # Create class map if it does not exist yet
            if not hasattr(self, "photon_class_map"):
                self.photon_class_map = make_tile_class_map(
                    holo.shape,
                    tile_size=getattr(self, "photon_tile_size", self.photon_tile_size),
                    n_classes=getattr(self, "photon_n_classes", self.photon_n_classes),
                    seed=getattr(self, "photon_class_seed", None),
                )

            # Create or regenerate kernel bank
            regenerate = getattr(self, "regenerate_photon_kernels", True)

            if regenerate or not hasattr(self, "photon_kernel_bank"):
                self.photon_kernel_bank = make_photon_kernel_bank(
                    n_classes=getattr(self, "photon_n_classes", self.photon_n_classes),
                    n_variants=getattr(self, "photon_n_variants", self.photon_n_variants),
                    size=getattr(self, "photon_kernel_size", self.photon_kernel_size),
                    sigma_range=getattr(
                        self,
                        "photon_sigma_range",
                        (0.5 * self.sigma_photon, 1.5 * self.sigma_photon),
                    ),
                    ellipticity_range=getattr(
                        self,
                        "photon_ellipticity_range",
                        (0.6, 1.6),
                    ),
                    irregularity=getattr(self, "photon_irregularity", self.photon_irregularity),
                    seed=getattr(self, "photon_kernel_seed", None),
                )

            photon_counts = photon_splat_with_spatial_classes(
                photon_counts=photon_counts,
                kernel_bank=self.photon_kernel_bank,
                class_map=self.photon_class_map,
                rng=rng,
            )


        # 9. convert photons back to detector counts
        holo = photon_counts * self.counts_per_photon
        holo = np.round(holo, 0)

        # 10. add gaussian readout noise from detector
        if (self.readout_noise_average > 0 or self.readout_noise_sigma > 0):
            holo += np.random.normal(
                self.readout_noise_average*self.number_frames,
                self.readout_noise_sigma*np.sqrt(self.number_frames),
                holo.shape)
            
        # 11 cap image at thresholding camera value
        holo=np.minimum(holo, self.number_frames*self.detector_threshold)

        # 12 divide by frame number: it is an average
        holo/= self.number_frames
            
        # just making sure the final product is positive
        holo[holo<0]=0

        self.hologram_exp=holo.copy()




    def add_noise_old(self):
        '''
        Given the hologram, the function simulates the holograms introducing drift,
         coherence effects and Poisson noise
        INPUT:
                readout_noise_average, readout_noise_sigma: readout noise of the camera
                sigma_h_px: sigma of drift in pixles. Takes into account also the spatial incoherence, so that has to be be taken ito accout too
                max_counts_per_image: max number of counts the camera can take in one image
                counts_per_photon:
                number_of_frames: number of acquired frames. The more, the lower the noise
                
        ----------
        Author: RB_2020
        '''

        holo=self.hologram_detector
        npx,npy=holo.shape

        # 4. simulate sample drift / vibrations AND SPATIAL INCOHERENCE
        if self.sigma_h_px > 0:
            kernel = np.outer(signal.windows.gaussian(npx, self.sigma_h_px), signal.windows.gaussian(npx, self.sigma_h_px))
            if kernel.sum() > 0:
                kernel /= kernel.sum()
                holo = signal.fftconvolve(holo,kernel,mode='same')

        # 6. adjust the maximum count to 64000 for a single image
        factor_holo=(self.max_counts_per_image)/np.amax((1.-self.beamstop.beamstop)*holo)
        holo*=factor_holo

        # consider you will have more than one frame
        holo *= self.number_frames        

        # 7. add Poisson noise to number of photons (sqrt(counts/counts_per_photon))
        holo /= self.counts_per_photon
        holo = np.random.poisson(holo).astype(np.float64)

        # 8. round to integer number of photons and 
        holo = np.round(holo,0)
        
        # 8.b I guess here we should have a SPF for a single photon
        if self.sigma_photon > 0:
            kernel = np.outer(signal.windows.gaussian(npx, self.sigma_photon), signal.windows.gaussian(npx, self.sigma_photon))
            if kernel.max() > 0:
                # Normalize by max to preserve peak values (not total signal)
                kernel /= kernel.max()
                holo = signal.fftconvolve(holo, kernel, mode='same')


        # 9. convert back to counts and re-rounding
        holo=holo*self.counts_per_photon

        holo = np.round(holo,0)

        # 10. add gaussian readout noise from detector
        if (self.readout_noise_average > 0 or self.readout_noise_sigma > 0):
            holo += np.random.normal(
                self.readout_noise_average*self.number_frames,
                self.readout_noise_sigma*np.sqrt(self.number_frames),
                holo.shape)
            
        # 11 cap image at thresholding camera value
        holo=np.minimum(holo, self.number_frames*self.detector_threshold)


        # 12 divide by frame number: it is an average
        holo/= self.number_frames
            
        # just making sure the final product is positive
        holo[holo<0]=0

        self.hologram_exp=holo.copy()


                        
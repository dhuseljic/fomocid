import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.ndimage import map_coordinates
from scattering_calculator.experimental_conditions.detector import detector_layout
from scattering_calculator.utils import image_transformator
from scipy.interpolate import griddata

class detector_hologram:
    def __init__(self, detector_layout, hologram, beam_parameters,real_space_pixel_size):
        self.detector_layout = detector_layout
        self.hologram = hologram
        self.beam_parameters = beam_parameters
        self.real_space_pixel_size=real_space_pixel_size
        self.extent_real = self.detector_layout.get_detector_extent_real_space()
        self.sample_shape = self.hologram.shape
        self.q_max_sim=np.pi / self.real_space_pixel_size


    def gnomonic_projection2(self) -> NDArray[np.float64]:
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
        print(Dq, Dq*self.hologram.shape[0], np.amax(detqx), np.amax(detqy))

        # we can use detx and dety to calculate the qx, qy coordinates in the far field corresponding to the real-space coordinates of the detector pixels
        # and then we can use these to decide where to sample hologram to have a gnomonic projection effect.
        # so, self.qxy are the q-coordinates the detector is actually mapping
        # while QX,QY are the perfect q coordinates of the sample simulation.
        # to get an actual hologram, we must map hologram(QX,QY) in the (detqx, detqy) coordinates
        #we do this using interpolation

        ## we just need to rescale detqx so they are expressed in absolute pixel value
        self.hologram_detector =map_coordinates(self.hologram, 
                                                [detqx/Dq*self.hologram.shape[0]+1*self.hologram.shape[0]/2,
                                                 detqy/Dq*self.hologram.shape[0]+1*self.hologram.shape[0]/2], 
                                                 order=3, mode='constant', cval=0)
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
        holo=self.hologram_detector

        # 4. simulate sample drift / vibrations AND SPATIAL INCOHERENCE
        if sigma_h_px > 0:
            kernel = np.outer(signal.gaussian(npx, sigma_h_px), signal.gaussian(npx, sigma_h_px))
            if kernel.sum() > 0:
                kernel /= kernel.sum()
                holo = signal.fftconvolve(holo,kernel,mode='same')

        # 6. adjust the maximum count to 64000 for a single image
        factor_holo=(max_counts_per_image)/holo.max()
        holo*=factor_holo

        # consider you will have more than one frame
        holo *= number_frames        

        # 7. add Poisson noise to number of photons (sqrt(counts/counts_per_photon))
        holo /= counts_per_photon
        holo = np.random.poisson(holo).astype(np.float64)


        # 8. round to integer number of photons and convert back to counts
        holo = np.round(holo,0)*counts_per_photon
        
        # 9. add gaussian readout noise from detector
        if (readout_noise_average > 0 or readout_noise_sigma > 0):
            holo += np.random.normal(readout_noise_average*number_frames,readout_noise_sigma*np.sqrt(number_frames),holo.shape)
            
        # 10 divide by frame number: it is an average
        holo/= number_frames
            
        # just making sure the final product is positive
        holo[holo<0]=0

        self.hologram_exp=holo.copy()

                        
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
        r = np.sqrt(detx**2 + dety**2 + z**2)
        center=self.detector_center
   
        

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
        self.hologram_detector =map_coordinates(self.hologram, [detqx/Dq,detqy/Dq], order=3, mode='constant', cval=0)
        # now we calculate the spatial coordinates on the detector plane corresponding to these qx, qy coordinates, 
        
    '''
    def gnomonic_projection(self) -> NDArray[np.float64]:
        Apply gnomonic projection to the hologram to correct for curvature of the Ewald sphere.
        Returns
        -------
            hologram_gnomonic : ndarray of shape (Ny, Nx)
            Gnomonic-projected hologram.        
        
        
        k = 2 * np.pi / self.beam_parameters.wavelength
        detx = self.detector_layout.detx
        dety = self.detector_layout.dety
        z = self.detector_layout.distance_sample_detector
        r = np.sqrt(detx**2 + dety**2 + z**2)
        center=self.detector_center
        
        method="cubic"
        mask=None
        
        values=self.hologram.flatten()
        points=np.array([dety.flatten(), detx.flatten()]).T.astype('float64')
            
        #now we have to calculate the new points
        points2=np.zeros(points.shape)
        points2[:,0]= z* np.sin( np.arctan( points[:,0] / np.sqrt( points[:,1] **2 + z**2 ) ) )
        points2[:,1]= z* np.sin( np.arctan( points[:,1] / np.sqrt( points[:,0] **2 + z**2 ) ) )

        
        CCD_projected = griddata(points2, values, points, method=method)
        
        CCD_projected = np.reshape(CCD_projected, self.detector_layout.detector_shape)
        
        #makes outside from nan to zero
        CCD_projected=np.nan_to_num(CCD_projected, nan=0, posinf=0, neginf=0)
        
        return CCD_projected
    
    '''



def simulHolo3(holo, detector,  Cntrs_level=1, readout_noise_average=0,
               readout_noise_sigma=3, sigma_h_px=0,
             max_counts_per_image=64000, counts_per_photon=50, number_frames=100, bs_size=0, sigma = 3, harm_factors=[]):
    
    '''
    Given two starting real space images (for the two helicities), the function simulates the holograms introducing drift,
    contrast levels, coherence effects and Poisson noise
    INPUT:
            im_p_focused,im_n_focused: positive and negative helicity images
            Cntrs_level: contrast of the image. Can make the contrast lower than the original for values <1
            readout_noise_average, readout_noise_sigma: readout noise of the camera
            sigma_h_px: sigma of drift in pixles. Takes into account also the spatial incoherence, so that has to be be taken ito accout too
            max_counts_per_image: max number of counts the camera can take in one image
            counts_per_photon:
            number_of_frames: number of acquired frames. The more, the lower the noise
            
    ----------
    Author: RB_2020
    '''
    

    npx,npy=sample_pos.shape

    # 1. the coherent hologram is given by |FFT(object+reference)|^2
    pos_ideal = np.abs(np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(sample_pos))))**2
    neg_ideal = np.abs(np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(sample_neg))))**2
    
    pos,neg=pos_ideal.copy(), neg_ideal.copy()
    


    
    
    # 4. simulate sample drift / vibrations AND SPATIAL INCOHERENCE
    if sigma_h_px > 0:
        kernel = np.outer(signal.gaussian(npx, sigma_h_px), signal.gaussian(npx, sigma_h_px))
        if kernel.sum() > 0:
            kernel /= kernel.sum()
            pos = signal.fftconvolve(pos,kernel,mode='same')
            neg = signal.fftconvolve(neg,kernel,mode='same')
            
    #mask beamstop
    pos=fth.mask_beamstop(pos, bs_size, sigma, center = None)
    neg=fth.mask_beamstop(neg, bs_size, sigma, center = None)

    # 6. adjust the maximum count to 64000 for a single image
    factor_pos=(max_counts_per_image)/pos.max()
    pos *=factor_pos 
    factor_neg=(max_counts_per_image)/neg.max()
    neg *= factor_neg
    # just renormalize the ideal hologram to match the non-ideal one
    pos_ideal*= factor_pos 
    neg_ideal*= factor_neg
    
    # consider you will have more than one frame
    pos *= number_frames
    neg *= number_frames
    
    
    # 7. add Poisson noise to number of photons (sqrt(counts/counts_per_photon))
    pos /= counts_per_photon
    neg /= counts_per_photon


    pos = np.random.poisson(pos).astype(np.float64)
    neg = np.random.poisson(neg).astype(np.float64)
    


    # 8. round to integer number of photons and convert back to counts
    pos = np.round(pos,0)
    neg = np.round(neg,0)
    pos *= counts_per_photon
    neg *= counts_per_photon
    
    # 9. add gaussian readout noise
    if (readout_noise_average > 0 or readout_noise_sigma > 0):
        pos += np.random.normal(readout_noise_average*number_frames,readout_noise_sigma*np.sqrt(number_frames),pos.shape)
        neg += np.random.normal(readout_noise_average*number_frames,readout_noise_sigma*np.sqrt(number_frames),neg.shape)
        
    pos/= number_frames
    neg/= number_frames
        
    # just making sure the final product is positive
    pos[pos<0]=0
    neg[neg<0]=0
    

    
    """
    # other things to do:
    - possibility to add a constant background, or low frequency background such as a more illuminated side, to try and model any stray light
    - add the BeamStop wires, the ones holding the stripe
    - the hologram is not always center. Uncenter it a bit randomly. Like 1/5 of camera misalignment
    - add interna modulations to the referece holes
    - add higher harmonics content
    - add cosmic rays
    - add hot/cold pixels
    
    """
    return pos,neg, pos_ideal, neg_ideal

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter



def Multislice(material=np.zeros((1000,1000,100),dtype=complex),field=np.ones((1000,1000),dtype=complex), l=1.59e-9, z=2050e-9, px_size=5e-9):
    '''
    Multislice simulations, computes transmission functions of a complex refractive index material matrix "material"
    of thickness "z" for a plane wave of wavelenght l
    can be used to compute transmittance of reference holes and object holes
    for object holes, field should be specified using the transmittance of the magnetized cobalt

    OUTPUT: field values at the end of the membrane
    (http://dx.doi.org/10.1364/OE.25.001831)
    RB_2020
    '''
    #first of all we wanna use an appropriate number of pixels, so that l**2*(ux**2+uy**2)) < 1 e quindi 2ux**-2<l**2 e quindi px_size>l/sqrt(2)
    # px size=hole_diam/npx e quindi hole_diam/npx>l/sqrt(2) e quindi npx<hole_diam*sqrt(2)/l
    #we have to reduce the resolution of the material matrix if it is too detailed. No

    npx=material.shape[0]
    npy=material.shape[1]
    n_layers=material.shape[2]

    #size of pixel
    #distance between layers
    dz = z/n_layers
    #propagator that will be used often
    prop = 1j*2*np.pi*dz/l
    #spatial frequencies
    Y,X = np.meshgrid(range(npx),range(npx))
    ux =  (X-npx/2.) * (1/px_size / (material.shape[0]//2))     # / ((npx/2.)*px_size)
    uy =  (Y-npy/2.) * (1/px_size / (material.shape[0]//2))     # / ((npy/2.)*px_size)

    for i in range(n_layers):
        #field enter ith slab and gets modified by material
        field *= np.exp( prop*(material[:,:,i]) )
        #field is propagated to next slab
        field  = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(field)))
        field *= np.exp( -prop * np.sqrt(1-l**2*(ux**2+uy**2)))
        field  = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(field)))

    return field

def Multislice_vacuum(field=np.ones((1000,1000),dtype=complex), l=1.59e-9, z=2050e-9, px_size=5e-9):
    '''
    Multislice simulations, plane wave field of l wavelengthis propagated in a space z
    can be used to compute transmittance of reference holes and object holes
    for object holes, field should be specified using the transmittance of the magnetized cobalt

    OUTPUT: field values at the end of the membrane
    (http://dx.doi.org/10.1364/OE.25.001831)
    RB_2020
    '''
    #first of all we wanna use an appropriate number of pixels, so that l**2*(ux**2+uy**2)) < 1 e quindi 2ux**-2<l**2 e quindi px_size>l/sqrt(2)
    # px size=hole_diam/npx e quindi hole_diam/npx>l/sqrt(2) e quindi npx<hole_diam*sqrt(2)/l
    #we have to reduce the resolution of the material matrix if it is too detailed. No

    npx=field.shape[0]
    npy=field.shape[1]
    prop_l0 = npx*px_size**2/l

    #propagator that will be used often
    prop = 1j*2*np.pi*z/l
    #spatial frequencies
    Y,X = np.meshgrid(np.arange(npx)-npx/2.,np.arange(npx)-npx/2.)
    ux =  X * (1/px_size / (npx / 2.))     # / ((npx/2.)*px_size)
    uy =  Y * (1/px_size / (npx / 2.))     # / ((npy/2.)*px_size)

    if z>prop_l0:
        #propagation
        field *= np.exp(-1j*np.pi/(l*z)*(X**2+Y**2)*px_size**2 )
        field  = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(field    )))
        field *= 1j/(l*z) * np.exp( -1j*np.pi*l*z*(ux**2+uy**2))

        # the coordinates are now (ux*l*dz, uy*l*z) rather than Y*px_size,X*px_size
        interp = RegularGridInterpolator(
            ( ux[:,0]*l*z, ux[:,0]*l*z ),
            #((np.arange(npx)-npx/2.)* (1./px_size / (field.shape[0]/2.))*l*z, (np.arange(npx)-npx/2.)* (1./px_size / (field.shape[0]/2.))*l*z),
            field,
            method='linear',
            bounds_error=False,
            fill_value=0 )

        # Stack them for interpolation input
        pts = np.stack([Y*px_size, X*px_size], axis=-1)
        # Evaluate interpolation
        field = interp(pts)

    else:
        #field is propagated to next slab
        field  = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(field)))
        field *= np.exp( -prop * np.sqrt(1-l**2*(ux**2+uy**2)))
        field  = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(field)))

    return field



def ellipse_coeffs(a, b,  A):
    # A in radians
    c0 = (np.cos(A)**2)/a**2 + (np.sin(A)**2)/b**2

    c1 = (np.sin(A)**2)/a**2 + (np.cos(A)**2)/b**2

    c2 =  np.sin(2*A)/a**2 - np.sin(2*A)/b**2

    c5 =  - 1

    return c0, c1, c2, c5

def randomify_binary(A, sigma_noise=5., alpha=555, lpx=1e-9, dz=1e-9):

    '''
    takes a binary volume and adds randomness to it. only works with binary volumes
    A:  binary volume defining the material: 1 = Au/Cr gold, 0 :hole
    sigma_noise: controls smoothness of bumps (larger = smoother, bigger features)
    alpha: controls how strong the surface deformation is
    '''

    mask = A.astype(bool)
    if A.ndim==3:
        sampling=(lpx,lpx,dz)
    if A.ndim==2:
        sampling=(lpx,lpx)

    # 1. Signed distance field: positive inside, negative outside
    dist_inside = distance_transform_edt(mask, sampling=sampling)
    dist_outside = distance_transform_edt(~mask, sampling=sampling)
    signed_dist = dist_inside - dist_outside   # >0 inside, <0 outside

    # 2. Smoothed random field
    rng = np.random.default_rng()
    noise = rng.normal(loc=0.0, scale=1, size=A.shape)
    noise_smooth = gaussian_filter(noise, sigma=sigma_noise)
    perturbed_field = signed_dist + alpha * noise_smooth

    # 3. Back to binary volume
    A = (perturbed_field > 0).astype(np.uint8)

    if False:
        # 4. make sure there are no negative concavities
        B=np.argmax(np.pad(A,  pad_width=((0,0), (0,0),(0,1))) ==0, axis=2)
        # Create an array of z-indices with shape (1,1,nz)
        z = np.arange(A.shape[2])[None, None, :]
        # Compare z with B, which has shape (nx,ny)
        A = (z < B[:, :, None]).astype(np.uint8)

    return A


def material_hole(hole_type     = 'RH',
                  lpx=10e-9,
                  Au_z             = 1850e-9,
                  dz            = 20e-9,
                  SiN_z         = 200e-9,
                  rx             = 50e-9,
                  ry             = 50e-9,
                  l             = 1.59e-9,
                  funnel_r            = 0.5e-6,
                  Df            = 2,
                  funnel_start  = 0.25,
                  ax=0,
                  ay=0,
                  az=0,
                  rotation=0,
                  sigma_noise=5.,
                  alpha_noise=100,
                  beta_mask=1.9474e-3, delta_mask=2.9474e-3,
                  delta_SiN= 0.00114742666, beta_SiN=  0.000194844441,
                  xh=0,yh=0,

                 ):
    '''
    fabricates obj/reference hole matrix
    the hole is always as big as the matrix, will be accomodated later
    keeps into account if it is an obj. hole or a ref. hole
    magnetic pattern should be ones for ref holes
    hole_diam, r_diam refer to the hole dimensions on the SiN side

    OUTPUT: the size of the pixel for the material + a complex 3D numpy array, containing the material refractive indexes in each voxel
    RB_2020
    '''
    #first of all we wanna use an appropriate number of pixels, so that l**2*(ux**2+uy**2)) < 1 e quindi 0.5*ux**-2<l**2 e quindi px_size>l/sqrt(0.5)
    # px size=hole_diam/npx e quindi hole_diam/npx>l/sqrt(0.5) e quindi npx<hole_diam*sqrt(0.5)/l

    alpha=ax*np.pi/180
    beta =ay*np.pi/180
    gamma=(az+rotation)*np.pi/180

    max_diam = np.amax([rx,ry,funnel_r])
    if hole_type=="slit":
        max_diam = np.amax([max_diam, rx+funnel_r,ry+funnel_r, np.sqrt((rx+funnel_r)**2+(ry+funnel_r)**2)])

    lpx = np.amax([1.05*(np.sqrt(2)*l),lpx])

    max_diam2 = Df*2*max_diam + (Au_z+SiN_z)*np.tan(alpha)

    npx=int(np.floor(  max_diam2 * np.cos(alpha) / lpx ))

    if npx%2==1:
        npx+=1
    npy=npx

    # we need to recalculate the size of the slab because of the tilt
    #total thickness of material / single layer
    t =  Df*2*max_diam * np.sin(alpha) + (Au_z+SiN_z)* (1/np.cos(alpha) + np.tan(alpha)*np.sin(alpha))


    npz=int(t/dz)
    dz= t/npz
    material=np.zeros((npy,npx,npz),dtype=complex)

    x0,y0,z0= npy/2*lpx, npx/2*lpx - np.sin(alpha)*(Au_z+SiN_z)/2 ,  t/2 - np.cos(alpha)*(Au_z+SiN_z)/2 -dz
    X,Y,Z=lpx*(np.arange(npx)) - x0, lpx*(np.arange(npy))-y0, dz*(np.arange(npz)) -z0
    X,Y,Z=np.meshgrid(X,Y,Z)



    Rx = np.array([[1, 0, 0],
                   [0, np.cos(alpha), -np.sin(alpha)],
                   [0, np.sin(alpha),  np.cos(alpha)]])

    Ry = np.array([[ np.cos(beta), 0, np.sin(beta)],
                   [0,            1, 0           ],
                   [-np.sin(beta), 0, np.cos(beta)]])

    Rz = np.array([[np.cos(gamma), -np.sin(gamma), 0],
                   [np.sin(gamma),  np.cos(gamma), 0],
                   [0,              0,             1]])

    coords = np.vstack([X.ravel(), Y.ravel(), Z.ravel()])   # (3, N)
    # rotate
    coords_rot = Rx @ coords                                 # (3, N)
    coords_rot = Rz @ coords_rot                                 # (3, N)

    # reshape back to original grid shape
    X = coords_rot[0].reshape(X.shape)
    Y = coords_rot[1].reshape(Y.shape)
    Z = coords_rot[2].reshape(Z.shape)


    M=np.zeros((npx,npy,npz))

    a,b,c=0,0.,1
    slab =  (a*X+b*Y+c*Z < Au_z+SiN_z)
    M[slab]=1


    if hole_type in ["OH", "RH"]:
        # Hole funnel
        radius = np.maximum(rx,ry) + (funnel_r-np.maximum(rx,ry)) /(Au_z*(1-funnel_start))* (Z-funnel_start*Au_z)
        radius = np.clip(radius, 0, None)
        cone = (X **2 + Y **2 <= radius**2)
        M[cone]=0

        # last cylinder
        #cyl = ((X / rx)**2 + (Y / ry)**2 <= 1)
        c0, c1, c2, c5=ellipse_coeffs(rx, ry, 0)
        cyl =  (c0*X**2 + c1*Y**2 + c2*X*Y + c5 )<=0
        M[cyl]=0

    if hole_type == "slit":
        slit_width  = rx + (funnel_r  ) /(Au_z*(1-funnel_start))* (Z-funnel_start*Au_z)
        slit_height = ry + (funnel_r  ) /(Au_z*(1-funnel_start))* (Z-funnel_start*Au_z)
        slit=(np.abs(X)<slit_width) & (np.abs(Y)<slit_height)
        M[slit]=0
        pass


    #randomify
    if sigma_noise!=0 or alpha_noise!=0:
        M=randomify_binary(M, sigma_noise=sigma_noise, alpha=alpha_noise, lpx=lpx, dz=dz)


    # SiN mask
    if hole_type=="OH":
        slab = (a*X+b*Y+c*Z < SiN_z ) & (a*X+b*Y+c*Z >= 0)
    else:
        slab = (a*X+b*Y+c*Z < SiN_z ) & (a*X+b*Y+c*Z >= 0) & (M==1)
    M[slab]=2

    # empty
    slab =  (a*X+b*Y+c*Z < 0)
    M[slab]=0

    M=M.astype(complex)
    #assign refractive index values
    M[M==1] = (delta_mask + 1j*beta_mask  )
    M[M==2] = (delta_SiN + 1j*beta_SiN  )

    return lpx,dz, M



import matplotlib.pyplot as plt

def hole_multislice(

                    hole_type     = 'RH',
                    lpx=10e-9,
                    Au_z             = 1850e-9,
                    dz            = 20e-9,
                    SiN_z         = 200e-9,
                    rx             = 50e-9,
                    ry             = 50e-9,
                    l             = 1.59e-9,
                    funnel_r            = 0.5e-6,
                    Df            = 2,
                    funnel_start  = 0.25,
                    ax=0,
                    ay=0,
                    az=0,
                    rotation=0,
                    sigma_noise=5.,
                    alpha_noise=100,
                    beta_mask=1.9474e-3,
                    delta_mask=2.9474e-3,
                    delta_SiN= 0.00114742666,
                    beta_SiN=  0.000194844441,
                    xh=0,
                    yh=0,
    plot_every=10
                    ):
    '''
    fabricates obj/reference hole matrix
    the hole is always as big as the matrix, will be accomodated later
    keeps into account if it is an obj. hole or a ref. hole
    magnetic pattern should be ones for ref holes
    hole_diam, r_diam refer to the hole dimensions on the SiN side

    OUTPUT: the size of the pixel for the material + a complex 3D numpy array, containing the material refractive indexes in each voxel
    RB_2020
    '''
    #first of all we wanna use an appropriate number of pixels, so that l**2*(ux**2+uy**2)) < 1 e quindi 0.5*ux**-2<l**2 e quindi px_size>l/sqrt(0.5)
    # px size=hole_diam/npx e quindi hole_diam/npx>l/sqrt(0.5) e quindi npx<hole_diam*sqrt(0.5)/l

    alpha=ax*np.pi/180
    beta =ay*np.pi/180
    gamma=(az+rotation)*np.pi/180

    max_diam = np.amax([rx,ry,funnel_r])
    if hole_type=="slit":
        max_diam = np.amax([max_diam, rx+funnel_r,ry+funnel_r, np.sqrt((rx+funnel_r)**2+(ry+funnel_r)**2)])

    lpx = np.amax([1.05*(np.sqrt(2)*l),lpx])

    max_diam2 = Df*2*max_diam + (Au_z+SiN_z)*np.tan(alpha)

    npx=int(np.floor(  max_diam2 * np.cos(alpha) / lpx ))

    if npx%2==1:
        npx+=1
    npy=npx

    # we need to recalculate the size of the slab because of the tilt
    #total thickness of material / single layer
    t =  Df*2*max_diam * np.sin(alpha) + (Au_z+SiN_z)* (1/np.cos(alpha) + np.tan(alpha)*np.sin(alpha))

    npz=int(t/dz)
    dz= t/npz
    material=np.zeros((npy,npx,npz),dtype=complex)

    x0,y0,z0= npy/2*lpx, npx/2*lpx - np.sin(alpha)*(Au_z+SiN_z)/2 ,  t/2 - np.cos(alpha)*(Au_z+SiN_z)/2 -dz


    x0,y0,z0= npy/2*lpx, npx/2*lpx  ,  0

    X,Y=lpx*(np.arange(npx)) - x0, lpx*(np.arange(npy))-y0
    X,Y=np.meshgrid(X,Y)
    #spatial frequencies

    ux =  (np.arange(npx)-npx/2) * (1/lpx / npx)
    uy =  (np.arange(npy)-npy/2) * (1/lpx / npy)
    ux,uy=np.meshgrid(ux,uy)

    #initial field
    field=np.ones((npx,npy),dtype=complex)

    #propagator that will be used often
    prop = 1j*2*np.pi*dz/l


    for i in range(npz):

        M=np.zeros((npx,npy))
        Z= X*0 + t- (i*dz - z0)

        ## first constructing the multislice material
        a,b,c=0,0.,1
        slab =  (a*X+b*Y+c*Z < Au_z+SiN_z)
        M[slab]=1



        if hole_type in ["OH", "RH"]:
            # Hole funnel
            radius = np.maximum(rx,ry) + (funnel_r-np.maximum(rx,ry)) /(Au_z*(1-funnel_start))* (Z-funnel_start*Au_z)
            radius = np.clip(radius, 0, None)
            cone = (X **2 + Y **2 <= radius**2)
            M[cone]=0

            # last cylinder
            #cyl = ((X / rx)**2 + (Y / ry)**2 <= 1)
            c0, c1, c2, c5=ellipse_coeffs(rx, ry, 0)
            cyl =  (c0*X**2 + c1*Y**2 + c2*X*Y + c5 )<=0
            M[cyl]=0

        if hole_type == "slit":
            slit_width  = rx + (funnel_r  ) /(Au_z*(1-funnel_start))* (Z-funnel_start*Au_z)
            slit_height = ry + (funnel_r  ) /(Au_z*(1-funnel_start))* (Z-funnel_start*Au_z)
            slit=(np.abs(X)<slit_width) & (np.abs(Y)<slit_height)
            M[slit]=0
            pass

        #randomify
        if sigma_noise!=0 or alpha_noise!=0:
            M=randomify_binary(M, sigma_noise=sigma_noise, alpha=alpha_noise, lpx=lpx, dz=dz)

        # SiN mask
        if hole_type=="OH":
            slab = (a*X+b*Y+c*Z < SiN_z ) & (a*X+b*Y+c*Z >= 0)
        else:
            slab = (a*X+b*Y+c*Z < SiN_z ) & (a*X+b*Y+c*Z >= 0) & (M==1)
        M[slab]=2

        # empty
        slab =  (a*X+b*Y+c*Z < 0)
        M[slab]=0



        M=M.astype(complex)
        #assign refractive index values
        M[M==1] = (delta_mask + 1j*beta_mask  )
        M[M==2] = (delta_SiN + 1j*beta_SiN  )

        ## then propagating
        #field enter ith slab and gets modified by material
        field *= np.exp( prop*(M[:,:]) )

        #field is propagated to next slab
        field  = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(field)))
        field *= np.exp( -prop * np.sqrt(1-l**2*(ux**2+uy**2)))
        field  = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(field)))

        if i%plot_every==0:
            fig,ax=plt.subplots()
            xx,yy=np.arange(npx),np.arange(npy)
            xx,yy=np.meshgrid(xx,yy)
            ax.imshow(np.abs(field), cmap="Spectral_r")
            ax.contour(xx,yy,np.abs(M) ,levels=1, colors="k", linewidths=1)

    return lpx, dz, field

'''
This file interfaces with the structcol package to simulate spectra.
'''

import numpy as np
import structcol as sc
from structcol import montecarlo as mc
import structcol.refractive_index as ri
from infer_structcol import main 

def calc_refl_trans(volume_fraction, Sample, ntrajectories=300, nevents=100, seed=None):
    """
    Calculates a reflection spectrum using the structcol package.

    Parameters
    ----------
    volume_fraction : float 
        volume fraction of scatterer in the system
    Sample : Sample object
        contains information about the sample that produced data
    ntrajectories : int
        number of trajectories
    nevents : int
        number of scattering events
    seed : int or None
        If seed is int, the simulation results will be reproducible. If seed is
        None, the simulation results are  random.
    
    Returns
    ----------
    reflection : ndarray
        fraction of reflected trajectories
    
    transmission : ndarray
        fraction of transmitted trajectories
        
    """
    # Read in system parameters from the Sample object
    particle_radius = sc.Quantity(Sample.particle_radius, 'nm')
    thickness = sc.Quantity(Sample.thickness, 'um')
    particle_index = sc.Quantity(Sample.particle_index, '')
    matrix_index = sc.Quantity(Sample.matrix_index, '')
    medium_index = sc.Quantity(Sample.medium_index, '')
    incident_angle = Sample.incident_angle
    wavelength = sc.Quantity(Sample.wavelength, 'nm')
    
    # Calculate the effective index of the sample
    sample_index = ri.n_eff(particle_index, matrix_index, volume_fraction)        
    
    reflectance = []
    transmittance = []
    for i in np.arange(len(wavelength)):    
        # Calculate the phase function and scattering and absorption lengths 
        # from the single scattering model
        p, mu_scat, mu_abs = mc.calc_scat(particle_radius, particle_index[i], 
                                          sample_index[i], volume_fraction, 
                                            wavelength[i], phase_mie=False, 
                                                mu_scat_mie=False)
            
        # Initialize the trajectories
        r0, k0, W0 = mc.initialize(nevents, ntrajectories, seed=seed, 
                                   incidence_angle=incident_angle)
        r0 = sc.Quantity(r0, 'um')
        k0 = sc.Quantity(k0, '')
        W0 = sc.Quantity(W0, '')

        # Generate a matrix of all the randomly sampled angles first 
        sintheta, costheta, sinphi, cosphi, _, _ = mc.sample_angles(nevents, ntrajectories, p)

        # Create step size distribution
        step = mc.sample_step(nevents, ntrajectories, mu_abs, mu_scat)
    
        # Create trajectories object
        trajectories = mc.Trajectory(r0, k0, W0, nevents)

        # Run photons
        trajectories.absorb(mu_abs, step)                         
        trajectories.scatter(sintheta, costheta, sinphi, cosphi)         
        trajectories.move(step)

        # Calculate the reflection fraction 
        R_fraction, T_fraction = mc.calc_refl_trans(trajectories, sc.Quantity('0.0 um'), 
                                        thickness, medium_index[i], 
                                        sample_index[i], detection_angle=np.pi/2)
        reflectance.append(R_fraction)
        transmittance.append(T_fraction)
        
    # Define an array for the visible wavelengths 
    wavelength_sigma = sc.Quantity(np.arange(400,1000,61), 'nm')
    # The uncertainty for the reflection fraction is taken to be 1 standard
    # deviation from the mean, and was calculated using the results of 100 identical runs.    
    sigma_measured = np.array([2.194931044327754974e-02, 2.550594209073478447e-02, 2.938120735169108197e-02, 
                               2.546565266246056738e-02, 2.766775254266129122e-02, 2.655298093906814996e-02, 
                               2.678825185819621799e-02, 2.581809869510796190e-02, 2.490913255664518877e-02, 
                               2.553807996587570059e-02, 2.628055161885621202e-02, 2.294961202628445496e-02, 
                               3.136882848529155138e-02, 2.760368719848924068e-02, 2.437879782412136556e-02, 
                               2.573038336586642993e-02, 2.442975389427453486e-02, 2.441993248675481884e-02, 
                               2.601376831467150375e-02, 2.396925947712269869e-02, 2.572289640994450399e-02, 
                               2.772440435733461786e-02, 2.798886114217479654e-02, 2.573708373567639221e-02, 
                               2.279423524456395539e-02, 2.402978393803960680e-02, 2.176312735840272433e-02, 
                               2.232057818335652158e-02, 1.862972739574862702e-02, 2.504705439142428797e-02,
                               2.179844061324090476e-02, 1.946989032181653187e-02, 2.020312176363358789e-02, 
                               2.029882527935277653e-02, 1.751614260502052864e-02, 1.879428814609657508e-02, 
                               1.656110381324961758e-02, 1.758803240888581407e-02, 1.715729536071330247e-02, 
                               1.453237765312232789e-02, 1.628458018940149785e-02, 1.483714489289071105e-02, 
                               1.536594722357430103e-02, 1.418281024003082390e-02, 1.454915622469696475e-02, 
                               1.362549922564000940e-02, 1.227806681320700935e-02, 1.294053179573567518e-02, 
                               1.405246509687570200e-02, 1.300492721566607474e-02, 1.194975110456097077e-02, 
                               1.280402054037179695e-02, 1.071386402168025484e-02, 1.142985119515856074e-02,
                               1.250892296358829987e-02, 1.195977451523276038e-02, 1.063376208175642948e-02, 
                               1.011335396883445738e-02, 1.071302028747354168e-02, 1.103299006390016240e-02, 
                               8.678388524720193065e-03])
                               
    # Find the uncertainties corresponding to each wavelength     
    wavelength_ind = main.find_close_indices(wavelength_sigma, wavelength)
    sigma = sigma_measured[np.array(wavelength_ind)]

    return main.Spectrum(wavelength.magnitude, reflectance = np.array(reflectance), 
                         transmittance = np.array(transmittance), sigma_r = sigma, sigma_t = sigma)

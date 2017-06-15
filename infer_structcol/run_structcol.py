'''
This file interfaces with the structcol package to simulate spectra.
'''

import numpy as np
import structcol as sc
from structcol import montecarlo as mc
import structcol.refractive_index as ri
from infer_structcol import main 

def calc_refl_trans(volume_fraction, radius, thickness, Sample, ntrajectories=600, nevents=200, seed=None):
    """
    Calculates a reflection spectrum using the structcol package.

    Parameters
    ----------
    volume_fraction : float 
        volume fraction of scatterer in the system
    radius : float (in nm)
        radius of scatterer
    thickness : float (in um)
        film thickness of sample
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
    particle_radius = sc.Quantity(radius, 'nm')
    thickness = sc.Quantity(thickness, 'um')
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
        r0, k0, W0 = mc.initialize(nevents, ntrajectories, medium_index[i], sample_index[i], seed=seed, 
                                   incidence_angle=incident_angle)
        r0 = sc.Quantity(r0, 'um')
        k0 = sc.Quantity(k0, '')
        W0 = sc.Quantity(W0, '')

        # Generate a matrix of all the randomly sampled angles first 
        sintheta, costheta, sinphi, cosphi, _, _ = mc.sample_angles(nevents, ntrajectories, p)

        # Create step size distribution
        step = mc.sample_step(nevents, ntrajectories, mu_abs, mu_scat)
    
        # Create trajectories object
        trajectories = mc.Trajectory(r0, k0, W0)

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
    wavelength_sigma = sc.Quantity(np.linspace(400,1000,61), 'nm')
    # The uncertainty for the reflection fraction is taken to be 1 standard
    # deviation from the mean, and was calculated using the results of 100 identical runs.    
    sigma_measured = np.array([1.860651421552072735e-02, 1.753980839818162357e-02, 1.839622738704549398e-02, 
                               1.596763386664768955e-02, 1.894484659740078986e-02, 1.722962247665738716e-02, 
                               1.555134197251030123e-02, 1.763293909648367200e-02, 2.027257609441594777e-02, 
                               1.850550125238413501e-02, 1.933699224240205058e-02, 1.873148138270526453e-02, 
                               1.908441182529240290e-02, 1.756355142274622708e-02, 1.590192651066632198e-02, 
                               1.596104976169695697e-02, 2.024553310180053287e-02, 1.955488448380025140e-02, 
                               1.882008022078682577e-02, 1.796507064336797313e-02, 2.004778422542081301e-02, 
                               1.811040666898488388e-02, 1.805909831464867776e-02, 1.810327867013098932e-02, 
                               1.516823124817042248e-02, 1.514314740128578328e-02, 1.696441336804245872e-02, 
                               1.677168419886158890e-02, 1.132382672347467811e-02, 1.224676407793331805e-02, 
                               1.117690246951372202e-02, 1.241312684961146107e-02, 1.326040920813134627e-02, 
                               1.367716094293736952e-02, 1.206014700075800326e-02, 1.250865278649789837e-02, 
                               1.060414384515132730e-02, 1.036736066347118505e-02, 9.217086600787521497e-03, 
                               1.110553603581512436e-02, 1.045612311627215976e-02, 9.731754980500811197e-03, 
                               9.142893085166936898e-03, 9.140217170604352653e-03, 8.149317528863142188e-03, 
                               7.833831850636231026e-03, 8.968239058921748802e-03, 7.848222412457096439e-03, 
                               7.609828884008617081e-03, 7.541218020424243600e-03, 7.964287577656070996e-03, 
                               7.665492573392837863e-03, 7.555737277343346943e-03, 7.034118091273018451e-03, 
                               6.271285385383303969e-03, 7.198737679024127048e-03, 5.980837995132812224e-03, 
                               6.166925243497538289e-03, 6.148309644049101616e-03, 6.087239500545048483e-03, 
                               6.549083556399931151e-03])
                                   
    # Find the uncertainties corresponding to each wavelength     
    wavelength_ind = main.find_close_indices(wavelength_sigma, wavelength)
    sigma = sigma_measured[np.array(wavelength_ind)]

    return main.Spectrum(wavelength.magnitude, reflectance = np.array(reflectance), 
                         transmittance = np.array(transmittance), sigma_r = sigma, sigma_t = sigma)

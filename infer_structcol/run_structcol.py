'''
This file interfaces with the structcol package to simulate spectra.
'''

import numpy as np
import structcol as sc
from structcol import montecarlo as mc
import structcol.refractive_index as ri
from infer_structcol import main 

def calc_refl_trans(volume_fraction, Sample, ntrajectories=300, nevents=200, seed=None):
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
    sigma_measured = np.array([1.578339786806479475e-02, 1.814049675099610806e-02, 2.263508305348480368e-02, 
                               2.280651893165159400e-02, 2.289441072296988580e-02, 2.475289930703982594e-02, 
                               2.591244161256863951e-02, 2.432751507610093206e-02, 2.840212103614853448e-02, 
                               2.464656608869982696e-02, 2.535837658100221007e-02, 2.352910256218017013e-02, 
                               2.264365724262535837e-02, 2.574192164180175851e-02, 2.546192844771695551e-02, 
                               2.767992948024671981e-02, 2.399941085043348632e-02, 2.767133759422578734e-02, 
                               2.759793344858079908e-02, 2.581248267951743655e-02, 2.664000919072649631e-02, 
                               2.914272756298553688e-02, 2.549173729396642454e-02, 2.722649681737301583e-02, 
                               2.322297011391676741e-02, 2.409138186086920430e-02, 2.807311239866464025e-02, 
                               3.018509924866123045e-02, 2.929772336148638717e-02, 2.866675231142475078e-02, 
                               2.377896176281297722e-02, 2.532538972626817778e-02, 2.408458082494839558e-02, 
                               2.823887112376391451e-02, 2.285680624758363796e-02, 2.834624619602043455e-02, 
                               2.342167374818072967e-02, 2.896504983742856365e-02, 2.835463183413225105e-02, 
                               2.981124596936481769e-02, 2.499991827718371987e-02, 2.697080309787770400e-02, 
                               2.788310424558666095e-02, 2.819362357263776805e-02, 2.852537757990830647e-02, 
                               2.651641629976883227e-02, 3.022850005391930842e-02, 2.772006618991802729e-02, 
                               2.971671988748269405e-02, 3.219841220549832933e-02, 2.570752641741474998e-02, 
                               2.352680291863861600e-02, 2.709648629442167056e-02, 2.524674214046034038e-02, 
                               2.758045644043585765e-02, 2.607698592177773098e-02, 2.738258841523178236e-02, 
                               2.868487596917410412e-02, 3.176931078488830218e-02, 2.729837088883461590e-02, 
                               2.513728413028934808e-02])
                                   
    # Find the uncertainties corresponding to each wavelength     
    wavelength_ind = main.find_close_indices(wavelength_sigma, wavelength)
    sigma = sigma_measured[np.array(wavelength_ind)]

    return main.Spectrum(wavelength.magnitude, reflectance = np.array(reflectance), 
                         transmittance = np.array(transmittance), sigma_r = sigma, sigma_t = sigma)

'''
This file interfaces with the structcol package to simulate spectra.
'''

import numpy as np
import structcol as sc
from structcol import montecarlo as mc
import structcol.refractive_index as ri
from infer_structcol import main 
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
import os
sns.set(font_scale=1.3) 

def calc_sigma(volume_fraction, radius, thickness, Sample, ntrajectories, nevents, run_num=100, plot=False, seed=None):
    """
    Calculates the standard deviation of the multiple scattering calculations
    by running the multiple scattering code run_num times.

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
        number of trajectories for the multiple scattering calculations
    nevents : int
        number of scattering events for the multiple scattering calculations
    run_num : int or 100
        number of runs from which to calculate the standard deviation
    plot : boolean 
        If True, plot of the theoretical reflectance and transmittance uncertainties
    seed : int or None
        If seed is int, the simulation results will be reproducible. If seed is
        None, the simulation results are  random.
    
    Returns
    ----------
    sigma_r : ndarray
        standard deviation of the reflectance calculation
    sigma_t : ndarray
        standard deviation of the transmittance calculation
        
    """   
    wavelength = sc.Quantity(Sample.wavelength, 'nm')
    
    reflectance = np.zeros([run_num, len(wavelength)])
    transmittance = np.zeros([run_num, len(wavelength)])
    for n in np.arange(run_num):
         reflectance[n,:], transmittance[n,:] = calc_refl_trans(volume_fraction, radius, thickness, 
                                                                Sample, ntrajectories, nevents, seed)

    # Calculate mean and standard deviations of reflectance and transmittance
    sigma_r = np.std(reflectance, axis=0) * np.sqrt(len(wavelength)/(len(wavelength)-1))
    sigma_t = np.std(transmittance, axis=0) * np.sqrt(len(wavelength)/(len(wavelength)-1))
    mean_r = np.mean(reflectance, axis=0)
    mean_t = np.mean(transmittance, axis=0)
    
    #np.savetxt(os.path.join(os.getcwd(),'sigma.txt'), np.array([sigma_R, sigma_T]).T)
    
    if plot == True:
        # Plot the mean and standard deviations
        fig, (ax_r, ax_t) = plt.subplots(2, figsize=(8,10))
        ax_r.set(ylabel='Reflectance')
        ax_t.set(ylabel='Transmittance')
        ax_t.set(xlabel='Wavelength (nm)')
        ax_r.errorbar(wavelength.magnitude, mean_r, yerr=sigma_r, fmt='.')
        ax_t.errorbar(wavelength.magnitude, mean_t, yerr=sigma_t, fmt='.')
        ax_r.set(title='Theoretical reflectance and transmittance +/- 1 standard deviation')

    return(sigma_r, sigma_t)
    

def calc_refl_trans(volume_fraction, radius, thickness, Sample, ntrajectories, nevents, seed):
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
        number of trajectories for the multiple scattering calculations
    nevents : int
        number of scattering events for the multiple scattering calculations
    seed : int or None
        If seed is int, the simulation results will be reproducible. If seed is
        None, the simulation results are  random.
    
    Returns
    ----------
    reflectance : ndarray
        fraction of reflected trajectories over the wavelength range
    transmittance : ndarray
        fraction of transmitted trajectories over the wavelength range
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
    
    reflectance = np.zeros(len(wavelength))
    transmittance = np.zeros(len(wavelength))
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
        reflectance[i], transmittance[i] = mc.calc_refl_trans(trajectories, sc.Quantity('0.0 um'), 
                                        thickness, medium_index[i], 
                                        sample_index[i], detection_angle=np.pi/2)

    return(reflectance, transmittance)

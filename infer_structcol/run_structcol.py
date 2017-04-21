'''
This file interfaces with the structcol package to simulate spectra.
'''

import numpy as np
import structcol as sc
from structcol import montecarlo as mc
import structcol.refractive_index as ri

from infer_structcol import main 

def calc_reflectance(volume_fraction, Sample, ntrajectories=300, nevents=100, seed=None):
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

    # Define scattering angles (a non-zero minimum angle is needed) 
    min_angle = 0.01            
    angles = sc.Quantity(np.linspace(min_angle,np.pi, 200), 'rad')   
    
    reflection = []
    for i in np.arange(len(wavelength)):    
        # Calculate the phase function and scattering and absorption lengths from the single scattering model
        p, lscat, labs = mc.calc_scat(particle_radius, particle_index[i], sample_index[i], volume_fraction, angles, wavelength[i], phase_mie=False, lscat_mie=False)
        
        mua = 1 / labs                               
        mus = 1 / lscat             

        # Initialize the trajectories
        r0, k0, W0 = mc.initialize(nevents, ntrajectories, seed=seed, incidence_angle=incident_angle)
        r0 = sc.Quantity(r0, 'um')
        k0 = sc.Quantity(k0, '')
        W0 = sc.Quantity(W0, '')

        # Generate a matrix of all the randomly sampled angles first 
        sintheta, costheta, sinphi, cosphi, theta, phi = mc.sample_angles(nevents, ntrajectories, p, angles)

        # Create step size distribution
        step = mc.sample_step(nevents, ntrajectories, mua, mus)
    
        # Create trajectories object
        trajectories = mc.Trajectory(r0, k0, W0, nevents)

        # Run photons
        trajectories.absorb(mua, step)                         
        trajectories.scatter(sintheta, costheta, sinphi, cosphi)         
        trajectories.move(step)

        #W = trajectories.weight    # we currently don't run systems with absorbers 
        k = trajectories.direction
        r = trajectories.position

        # Calculate the reflection fraction 
        R_fraction = mc.calc_reflection(r[2], sc.Quantity('0.0 um'), thickness, ntrajectories, matrix_index[i], sample_index[i], k[0], k[1], k[2], detection_angle=np.pi/2)
        reflection.append(R_fraction)
        
    # Uncertainties are 1 standard deviation of 100 typical runs
    wavelength_sigma = sc.Quantity(np.arange(450.0,810.0,10.0), 'nm')        
    sigma_measured = np.array([2.931712774077597020e-02, 2.732862042239197695e-02, 2.372715146455805016e-02, 
             2.474964384227144529e-02, 2.745171731244393926e-02, 2.617463528791452027e-02,
             2.328581430854679071e-02, 2.487518756643465265e-02, 2.669051976970928178e-02,
             3.188603288730038066e-02, 2.574696936015777662e-02, 2.677392781333740729e-02, 
             2.817717959749582030e-02, 2.470622973847584267e-02, 2.672016007102775287e-02, 
             2.478513301958401929e-02, 2.547285076117726879e-02, 2.409074600405497826e-02, 
             2.261673006300473493e-02, 2.317995318227433418e-02, 2.259580668871176076e-02, 
             1.865577625363827596e-02, 1.978859946313229340e-02, 2.296197801587663428e-02, 
             2.315551804759025806e-02, 2.224703426828641625e-02, 1.985123936379590281e-02, 
             1.878124040756953134e-02, 1.771680400857145524e-02, 1.918301705919833039e-02, 
             1.586503561500365519e-02, 1.609805223343139732e-02, 1.699810700990839196e-02, 
             1.916387924804048917e-02, 1.512936623153174689e-02, 1.558802067101308815e-0])
    
    # Find the uncertainties corresponding to each wavelength     
    wavelength_ind = main.find_close_indices(wavelength_sigma, wavelength)
    sigma = sigma_measured[np.array(wavelength_ind)]

    return main.Spectrum(wavelength.magnitude, np.array(reflection), sigma)

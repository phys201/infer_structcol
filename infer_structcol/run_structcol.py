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
    wavelength_sigma = sc.Quantity(np.arange(400,1000,61), 'nm')
    # The uncertainty for the reflection fraction is taken to be 1 standard
    # deviation from the mean, and was calculated using the results of 100 identical runs.    
    sigma_measured = np.array([1.552158348290047334e-02, 1.529827850362531529e-02, 1.974236746672980783e-02, 
                               1.864905148599750595e-02, 2.107284819791623715e-02, 2.964145063023707061e-02, 
                               2.076492899997999658e-02, 2.463638014400696893e-02, 2.714664106184601008e-02, 
                               2.401441506019916933e-02, 2.291078818817240265e-02, 2.320253146559166091e-02, 
                               2.566265765495753109e-02, 2.115432469842693647e-02, 2.514050406586916878e-02, 
                               2.880597982387037709e-02, 2.894459734053560218e-02, 2.735528761454724886e-02, 
                               2.800905227182482957e-02, 2.524757733083715303e-02, 2.809012740946859737e-02, 
                               2.512416954348656653e-02, 2.467613195146483446e-02, 3.187343480325866008e-02, 
                               2.739911086347164570e-02, 2.500912108238710629e-02, 2.471610364649903108e-02, 
                               2.762659718492080363e-02, 2.803704484238538422e-02, 2.456886590028163272e-02, 
                               2.634529613950989044e-02, 2.956853998903598638e-02, 2.829313104337189202e-02, 
                               2.800036384250904897e-02, 2.839651317871575570e-02, 3.016367468349833922e-02, 
                               2.479798742434528858e-02, 2.720992601205637906e-02, 2.745665122968573083e-02, 
                               2.743295948690166811e-02, 2.618194666221104749e-02, 2.747720878472238032e-02, 
                               2.759432186139849225e-02, 2.776259934452364117e-02, 2.940350745602579932e-02, 
                               2.891366379419401181e-02, 2.771658542466151273e-02, 2.666780686899664771e-02, 
                               2.443015281339374484e-02, 2.849886563990108868e-02, 2.965280684939339273e-02, 
                               2.416725551568865704e-02, 2.547232583311598644e-02, 2.884550369351878210e-02, 
                               2.746804370084566030e-02, 2.537825409011371797e-02, 2.673418611005524623e-02, 
                               2.791063901458219060e-02, 2.673438912565991032e-02, 3.003603473997697171e-02, 
                               2.541179063043090200e-02])
                                   
    # Find the uncertainties corresponding to each wavelength     
    wavelength_ind = main.find_close_indices(wavelength_sigma, wavelength)
    sigma = sigma_measured[np.array(wavelength_ind)]

    return main.Spectrum(wavelength.magnitude, reflectance = np.array(reflectance), 
                         transmittance = np.array(transmittance), sigma_r = sigma, sigma_t = sigma)

import numpy as np
from . import structcol as sc
from structcol import montecarlo as mc
import structcol.refractive_index as ri

def calc_reflection(volume_fraction, Sample, ntrajectories=300, nevents=100):
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
    
    Returns
    ----------
    reflection : ndarray
        fraction of reflected trajectories
        
    """
    # Read in system parameters from the Sample object
    particle_size = Sample.particle_size
    thickness= Sample.thickness
    particle_index = Sample.particle_index
    matrix_index = Sample.matrix_index
    medium_index = Sample.medium_index
    incident_angle = Sample.incident_angle
    wavelength = Sample.wavelength
    
    # Calculate the effective index of the sample
    sample_index = ri.n_eff(particle_index, matrix_index, volume_fraction)      

    # Define scattering angles (a non-zero minimum angle is needed) 
    min_angle = 0.01            
    angles = sc.Quantity(np.linspace(min_angle,np.pi, 200), 'rad')   
        
    i = 0
    reflection = []
    for wavelen in wavelength:
        # Calculate the phase function and scattering and absorption lengths from the single scattering model
        p, lscat, labs = mc.calc_scat(particle_size, particle_index[i], sample_index[i], volume_fraction, angles, wavelen, phase_mie=False, lscat_mie=False)
        
        mua = 1 / labs                               
        mus = 1 / lscat             

        # Initialize the trajectories
        r0, k0, W0 = mc.initialize(nevents, ntrajectories, incidence_angle=incident_angle)
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
        R_fraction = mc.calc_reflection(r[2], sc.Quantity('0.0 um') , thickness, ntrajectories, matrix_index, sample_index, k[0], k[1], k[2], detection_angle=np.pi/2)
        reflection.append(R_fraction)
        i = i + 1
        
    return np.array(reflection)
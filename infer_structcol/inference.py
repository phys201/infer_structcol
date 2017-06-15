'''
This file contains the inference framework to estimate sample properties.
'''

import multiprocessing as mp
import numpy as np
import pandas as pd
import emcee
import lmfit
from .model import (calc_model_spect, calc_resid_spect, min_phi, max_phi, 
                    min_radius, max_radius, min_thickness, max_thickness, 
                    min_l0, max_l0, min_l1, max_l1, log_posterior)

def find_max_like(data, sample, theta_guess, theta_range, seed=None):
    '''
    Uses lmfit to approximate the highest likelihood parameter values.
    We use likelihood instead of posterior because lmfit requires an array of 
    residuals.

    Parameters
    ---------
    data: Spectrum object
        experimental dataset
    sample: Sample object
        information about the sample that produced data_spectrum
    seed: int (optional)
        if specified, passes the seed through to the MC multiple scattering 
        calculation
    theta_guess: list of floats (optional)
        user's best guess of the expected parameter values (phi, radius, 
        thickness)
    theta_range: 2 by 3 array of floats 
        user's best guess of the expected ranges of the parameter values 
        ([[min_phi, max_phi], [min_radius, max_radius], [min_thickness, max_thickness]]) 
    
    Returns
    -------
    theta: 5 or 7 -tuple
        best fit (phi, radius, thickness, l0, l1) or (phi, radius, thickness, 
        l0, l1, l0, l1) as floats
    '''
    if theta_guess == None:
        phi_guess = 0.55
        radius_guess = 120.
        thickness_guess = 120.
    else:
        phi_guess = theta_guess[0]
        radius_guess = theta_guess[1]
        thickness_guess = theta_guess[2]
    
    def resid(params):
        if 'reflectance' in data.keys() and 'transmittance' in data.keys():
            theta = (params['phi'], params['radius'],  params['thickness'], 
                     params['l0_r'], params['l1_r'], params['l0_t'], params['l1_t'])
        elif 'reflectance' in data.keys():
            theta = (params['phi'], params['radius'],  params['thickness'], 
                     params['l0_r'], params['l1_r'])
        else:
            theta = (params['phi'], params['radius'],  params['thickness'],
                     params['l0_t'], params['l1_t'])

        theory_spect = calc_model_spect(sample, theta, seed)
        resid_spect = calc_resid_spect(data, theory_spect)

        resid = np.concatenate([resid_spect.reflectance/resid_spect.sigma_r, 
                                resid_spect.transmittance/resid_spect.sigma_t])
        return resid[np.isfinite(resid)]

    fit_params = lmfit.Parameters()
    fit_params['phi'] = lmfit.Parameter(value=phi_guess, min=theta_range[0,0], max=theta_range[0,1])
    fit_params['radius'] = lmfit.Parameter(value=radius_guess, min=theta_range[1,0], max=theta_range[1,1])
    fit_params['thickness'] = lmfit.Parameter(value=thickness_guess, min=theta_range[2,0], max=theta_range[2,1])
    
    if 'reflectance' in data.keys():
        fit_params['l0_r'] = lmfit.Parameter(value=0.02, min=min_l0, max=max_l0)
        fit_params['l1_r'] = lmfit.Parameter(value=0., min=min_l1, max=max_l1)

    if 'transmittance' in data.keys(): 
        fit_params['l0_t'] = lmfit.Parameter(value=0.02, min=min_l0, max=max_l0)
        fit_params['l1_t'] = lmfit.Parameter(value=0., min=min_l1, max=max_l1)
        
    fit_params = lmfit.minimize(resid, fit_params).params
    return tuple(fit_params.valuesdict().values())


def run_mcmc(data, sample, nwalkers, nsteps, theta_guess = None, theta_range = None, seed=None):
    '''
    Performs actual mcmc calculation. Returns an Emcee Sampler object. 

    Parameters
    -------
    data: Spectrum object
        experimental dataset
    sample: Sample object
        information about the sample that produced data_spectrum
    nwalkers: int (even)
        number of parallelized MCMC walkers to use
    nsteps: int
        number of steps taken by each walker
    theta_guess: list of floats (optional)
        user's best guess of the expected parameter values (phi, radius, 
        thickness)
    theta_range: 2 by 3 array of floats (optional)
        user's best guess of the expected ranges of the parameter values 
        ([[min_phi, max_phi], [min_radius, max_radius], [min_thickness, max_thickness]]) 
    seed: int (optional)
        sets the seed for all MC scattering trajectory chains. 
        DOES NOT set the seed for MCMC walkers.
    '''    

    # set expected values to initialize walkers
    #if theta is None:
    #    theta = find_max_like(data, sample, seed)
    if theta_range == None:
        theta_range = np.array([[min_phi, max_phi], [min_radius, max_radius], 
                                [min_thickness, max_thickness]])
    
    theta = find_max_like(data, sample, theta_guess, theta_range, seed)
    
    ndim = len(theta)
      
    # set walkers in a distribution with width .05
    vf = np.clip(theta[0]*np.ones(nwalkers) + 0.01*np.random.randn(nwalkers), 
                 theta_range[0,0], theta_range[0,1])
    radius = np.clip(theta[1]*np.ones(nwalkers) + 1*np.random.randn(nwalkers), 
                 theta_range[1,0], theta_range[1,1])
    thickness = np.clip(theta[2]*np.ones(nwalkers) + 1*np.random.randn(nwalkers), 
                 theta_range[2,0], theta_range[2,1])
    l0 = np.clip(theta[3]*np.ones(nwalkers) + 0.01*np.random.randn(nwalkers), 
                 min_l0, max_l0)
    # clip the sum of l0 and l1 and subtract l0 to ensure that the clipped 
    # l1 satisfies 0 < l0 + l1 < 1        
    l1 = np.clip((theta[3] + theta[4]) * np.ones(nwalkers) + 
                0.01*np.random.randn(nwalkers), min_l1, max_l1) - l0
    
    if ndim == 5:
        theta = np.vstack((vf,radius,thickness,l0,l1)).T.tolist()

    if ndim == 7:
        l0_1 = np.clip(theta[5] * np.ones(nwalkers) + 
                       0.01*np.random.randn(nwalkers), min_l0, max_l0)
        l1_1 = np.clip((theta[5] + theta[6]) * np.ones(nwalkers) + 
                       0.01*np.random.randn(nwalkers), min_l1, max_l1) - l0_1
        theta = np.vstack((vf,radius,thickness,l0,l1,l0_1,l1_1)).T.tolist()
    
    #theta = [list(theta) + 0.05*np.random.randn(ndim) for i in range(nwalkers)]

    # figure out how many threads to use
    nthreads = np.min([nwalkers, mp.cpu_count()])

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, 
                                    args=[data, sample, theta_range, seed], threads=nthreads)
    sampler.run_mcmc(theta, nsteps)

    return sampler
    

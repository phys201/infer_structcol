'''
This file contains the inference framework to estimate sample properties.
'''

import multiprocessing as mp
import numpy as np
import pandas as pd
import emcee
import lmfit
from .model import (calc_model_spect, calc_resid_spect, min_phi, max_phi, min_l0, 
max_l0, min_l1, max_l1, log_posterior)

def find_max_like(data, sample, seed=None):
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
    phi_guess: float
        user's best guess of the expected volume fraction
    seed: int (optional)
        if specified, passes the seed through to the MC multiple scatterin 
        calculation

    Returns
    -------
    theta: 3 or 5 -tuple
        best fit (phi, l0, l1) or (phi, l0, l1, l0, l1) as floats
    '''
    def resid(params):
        if 'reflectance' in data.keys() and 'transmittance' in data.keys():
            theta = (params['phi'], params['l0_r'], params['l1_r'], 
                     params['l0_t'], params['l1_t'])
        elif 'reflectance' in data.keys():
            theta = (params['phi'], params['l0_r'], params['l1_r'])
        else:
            theta = (params['phi'], params['l0_t'], params['l1_t'])

        theory_spect = calc_model_spect(sample, theta, seed)
        resid_spect = calc_resid_spect(data, theory_spect)

        resid = np.concatenate([resid_spect.reflectance/resid_spect.sigma_r, 
                                resid_spect.transmittance/resid_spect.sigma_t])
        return resid[np.isfinite(resid)]

    fit_params = lmfit.Parameters()
    fit_params['phi'] = lmfit.Parameter(value=.55, min=min_phi, max=max_phi)
        
    if 'reflectance' in data.keys():
        fit_params['l0_r'] = lmfit.Parameter(value=0, min=min_l0, max=max_l0)
        fit_params['l1_r'] = lmfit.Parameter(value=0, min=min_l1, max=max_l1)

    if 'transmittance' in data.keys(): 
        fit_params['l0_t'] = lmfit.Parameter(value=0, min=min_l0, max=max_l0)
        fit_params['l1_t'] = lmfit.Parameter(value=0, min=min_l1, max=max_l1)
        
    fit_params = lmfit.minimize(resid, fit_params).params
    return tuple(fit_params.valuesdict().values())

def run_mcmc(data, sample, nwalkers, nsteps, theta = None, seed=None):
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
    theta: 3 or 5-tuple of floats (optional)
        user's best guess of the expected parameter values (phi, l0, l1) or 
        (phi, l0, l1, l0, l1)
    seed: int (optional)
        sets the seed for all MC scattering trajectory chains. 
        DOES NOT set the seed for MCMC walkers.
    '''    

    # set expected values to initialize walkers
    if theta is None:
        theta = find_max_like(data, sample, seed)

    ndim = len(theta)
    
    # set walkers in a distribution with width .05
    vf = np.clip(theta[0]*np.ones(nwalkers) + 0.05*np.random.randn(nwalkers), 
                 min_phi, max_phi)
    l0 = np.clip(theta[1]*np.ones(nwalkers) + 0.05*np.random.randn(nwalkers), 
                 min_l0, max_l0)
    l1 = np.clip(theta[2]*np.ones(nwalkers) + 0.05*np.random.randn(nwalkers), 
                 min_l1, max_l1)
    if ndim == 3:
        theta = np.vstack((vf,l0,l1)).T.tolist()
    if ndim == 5:
        l0_1 = np.clip(theta[3]*np.ones(nwalkers) + 
                       0.05*np.random.randn(nwalkers), min_l0, max_l0)
        l1_1 = np.clip(theta[4]*np.ones(nwalkers) + 
                       0.05*np.random.randn(nwalkers), min_l1, max_l1)
        theta = np.vstack((vf,l0,l1,l0_1,l1_1)).T.tolist()
    #theta = [list(theta) + 0.05*np.random.randn(ndim) for i in range(nwalkers)]

    # figure out how many threads to use
    nthreads = np.min([nwalkers, mp.cpu_count()])

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, 
                                    args=[data, sample, seed], threads=nthreads)
    sampler.run_mcmc(theta, nsteps)
    return sampler


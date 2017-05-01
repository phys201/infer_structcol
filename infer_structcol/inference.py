'''
This file contains the inference framework to estimate sample properties.
'''

import multiprocessing as mp
import numpy as np
import pandas as pd
import emcee
import lmfit

from .model import calc_model_spect, calc_resid_spect, min_phi, max_phi, log_posterior

def find_max_like(data, sample, seed=None):
    '''
    Uses lmfit to approximate the highest likelihood parameter values.
    We use likelihood instead of posterior because lmfit requires an array of residuals.

    Prameters
    ---------
    data: Spectrum object
        experimental dataset
    sample: Sample object
        information about the sample that produced data_spectrum
    phi_guess: float
        user's best guess of the expected volume fraction
    seed: int (optional)
        if specified, passes the seed through to the MC multiple scatterin calculation

    Returns
    -------
    theta: 3 or 5 -tuple
        best fit (phi, l0, l1) as floats
    '''
    def resid(params):
        if 'reflectance' in data.keys() and 'transmittance' in data.keys():
            theta = (params['phi'], params['l0_r'], params['l1_r'], params['l0_t'], params['l1_t'])
        elif 'reflectance' in data.keys():
            theta = (params['phi'], params['l0_r'], params['l1_r'])
        else:
            theta = (params['phi'], params['l0_t'], params['l1_t'])
        theory_spect = calc_model_spect(sample, theta, seed)
        resid_spect = calc_resid_spect(data, theory_spect)
        return resid_spect.reflectance/resid_spect.sigma_r

    fit_params = lmfit.Parameters()
    fit_params['phi'] = lmfit.Parameter(value=.55, min=min_phi, max=max_phi)
        
    if 'reflectance' in data.keys():
        fit_params['l0_r'] = lmfit.Parameter(value=0, min=0, max=1)
        fit_params['l1_r'] = lmfit.Parameter(value=0, min=-1, max=1)

    if 'transmittance' in data.keys(): 
        fit_params['l0_t'] = lmfit.Parameter(value=0, min=0, max=1)
        fit_params['l1_t'] = lmfit.Parameter(value=0, min=-1, max=1)
        
    fit_params = lmfit.minimize(resid, fit_params).params
    return (fit_params.valuesdict().values())

def get_distribution(data, sample, nwalkers=50, nsteps=500, burn_in_time=0, phi_guess = 0.55):
    '''
    Calls run_mcmc and outputs pandas DataFrame of parameters and log-probability
    
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
    burn_in_time: int
        number of inital steps to be removed from the returned samples
    phi_guess: float
        user's best guess of the expected volume fraction
    '''    

    walkers = run_mcmc(data, sample, nwalkers, nsteps, phi_guess)
    ndim = data.shape[1] # number of parameters happens to be equal to number of columns in data
    traces = np.concatenate([walkers.chain[:,burn_in_time:,:], walkers.lnprobability[:,burn_in_time:,np.newaxis]],axis=2).reshape(-1, ndim+1).T
    params = ['vol_frac']
    if 'reflectance' in data.keys():
        params.append('l0_r','l1_r')
    if 'transmittance' in data.keys():
        params.append('l0_r','l1_t')
    params.append('lnprob')
    return pd.DataFrame({key: traces[val] for val, key in enumerate(params)})

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
        user's best guess of the expected parameter values
    seed: int (optional)
        sets the seed for all MC scattering trajectory chains. 
        DOES NOT set the seed for MCMC walkers.
    '''    

    # set expected values to initialize walkers
    if theta is None:
        theta = find_max_like(data, sample, seed)

    ndim = len(theta)

    # set walkers in a distribution with width .05
    theta = [theta + 0.05*np.random.randn(ndim) for i in range(nwalkers)]

    # figure out how many threads to use
    nthreads = np.min([nwalkers, mp.cpu_count()])

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=[data, sample, seed], threads=nthreads)
    sampler.run_mcmc(theta, nsteps)
    return sampler


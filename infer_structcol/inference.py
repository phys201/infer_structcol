'''
This file contains the inference framework to estimate sample properties.
'''

import multiprocessing as mp
import numpy as np
import pandas as pd
import emcee
import lmfit

from .model import calc_model_spect, calc_resid_spect, min_phi, max_phi, log_posterior
from .run_structcol import calc_refl_trans

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
    theta: 3-tuple
        best fit (phi, l0, l1) as floats
    '''
    def resid(params):
        theta = (params['phi'], params['l0'], params['l1'])
        theory_spect = calc_model_spect(sample, theta, seed)
        resid_spect = calc_resid_spect(data, theory_spect)
        return resid_spect.reflectance/resid_spect.sigma_r

    fit_params = lmfit.Parameters()
    fit_params['l0'] = lmfit.Parameter(value=0, min=0, max=1)
    fit_params['l1'] = lmfit.Parameter(value=0, min=-1, max=1)
    fit_params['phi'] = lmfit.Parameter(value=.55, min=min_phi, max=max_phi)
    fit_params = lmfit.minimize(resid, fit_params).params
    return (fit_params['phi'].value, fit_params['l0'].value, fit_params['l1'].value)

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
    traces = np.concatenate([sampler.chain[:,burn_in_time:,:], sampler.lnprobability[:,burn_in_time:,np.newaxis]],axis=2).reshape(-1, ndim+1).T
    return pd.DataFrame({key: traces[val] for val, key in enumerate(['vol_frac','l0','l1','lnprob'])})

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
    theta: 3-tuple of floats (optional)
        user's best guess of the expected parameter values (phi, l0, l1) 
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


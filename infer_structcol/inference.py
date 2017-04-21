'''
This file contains the inference framework to estimate sample properties.
'''

import multiprocessing as mp
import numpy as np
import pandas as pd
import emcee
import lmfit

from .model import calc_resid_spect, min_phi, max_phi, log_posterior
from .run_structcol import calc_reflectance

def find_max_like(data, sample, phi_guess=0.55, seed=None):
    '''
    UNDER CONSTRUCTION
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
    '''
    def resid(params):
        theory_spect = calc_reflectance(params['phi'], sample, seed=seed)
        resid_spect = calc_resid_spect(data, theory_spect, params['l0'], params['l1'])
        print(np.sum(resid_spect.reflectance**2/resid_spect.sigma_r**2))
        return resid_spect.reflectance/resid_spect.sigma_r

    fit_params = lmfit.Parameters()
    fit_params['l0'] = lmfit.Parameter(value=0.5, min=0, max=1)
    fit_params['l1'] = lmfit.Parameter(value=0, min=-1, max=1)
    fit_params['phi'] = lmfit.Parameter(value=phi_guess, min=min_phi, max=max_phi)
    return lmfit.minimize(resid, fit_params)

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

def run_mcmc(data, sample, nwalkers, nsteps, phi_guess, seed=None):
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
    phi_guess: float
        user's best guess of the expected voluem fraction
    seed: int (optional)
        sets the seed for all MC scattering trajectory chains. 
        DOES NOT set the seed for MCMC walkers.
    '''    
    # set expected values to initialize walkers
    expected_vals = np.array([phi_guess,.5,0])
    ndim = len(expected_vals)
    
    # set walkers in a distribution with width .05
    starting_positions = [expected_vals + 0.05*np.random.randn(ndim) for i in range(nwalkers)]
    
    # figure out how many threads to use
    nthreads = np.min([nwalkers, mp.cpu_count()])

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=[data, sample, phi_guess, seed], threads=nthreads)
    sampler.run_mcmc(starting_positions, nsteps)
    return sampler


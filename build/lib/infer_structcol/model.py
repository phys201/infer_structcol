'''
This file contains the generative model and inference framework to calculate posterior probabilities.
'''

import multiprocessing as mp
import numpy as np
import pandas as pd
import emcee
from .main import rescale
from .run_structcol import calc_reflectance

def calc_log_prior(theta, phi_guess):
    '''
    Calculats log of prior probability of obtaining theta.
    
    Parameters
    -------
    theta: 3-tuple 
        set of inference parameter values - volume fraction, baseline loss, wavelength dependent loss
    phi_guess: float
        user's best guess of the expected voluem fraction
    '''
    vol_frac, l0, l1 = theta

    if l0 < 0 or l0 > 1 or l0+l1 <0 or l0+l1 > 1:
        # Losses are not in range [0,1] for some wavelength
        return -np.inf 

    if vol_frac < .35 or vol_frac > .73:
        # Outside range of validity of multiple scattering model
        return -np.inf
    
    var = .0025 # based on expected normal range
    return -(vol_frac-phi_guess)**2/var

def calc_likelihood(data, theory, l0, l1):
    '''
    Returns likelihood of obtaining an experimental dataset from a given theoretical spectrum
    
    Parameters
    -------
    data: Spectrum object
        experimental dataset
    theory: Spectrum object
        calculated dataset
    l0: float
        baseline loss parameter
    l1: float
        wavelength-dependent loss parameter
    '''
    loss = l0 + l1*rescale(data.wavelength)
    residual = data.reflectance - (1-loss)*theory.reflectance
    var_eff = data.sigma_r**2 + ((1-loss)*theory.sigma_r)**2
    chi_square = np.sum(residual**2/var_eff)
    prefactor = 1/np.prod(np.sqrt(2*np.pi*var_eff))
    return prefactor * np.exp(-chi_square/2)

def log_posterior(theta, data_spectrum, sample, phi_guess, seed=None):
    '''
    Calculates log-posterior of a set of parameters producing an observed reflectance spectrum
    
    Parameters
    -------
    theta: 3-tuple 
        set of inference parameter values - volume fraction, baseline loss, wavelength dependent loss
    data_spectrum: Spectrum object
        experimental dataset
    sample: Sample object
        information about the sample that produced data_spectrum
    phi_guess: float
        user's best guess of the expected voluem fraction
    seed: int (optional)
        if specified, passes the seed through to the MC multiple scatterin calculation
    '''    
    vol_frac, l0, l1 = theta
    if not np.all(sample.wavelength == data_spectrum.wavelength):
        raise ValueError("Sample and data must share the same set of wavelengths")
    theory_spectrum = calc_reflectance(vol_frac, sample, seed=seed)

    likelihood = calc_likelihood(data_spectrum, theory_spectrum, l0, l1)
    return np.log(likelihood) + calc_log_prior(theta, phi_guess)

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
        user's best guess of the expected voluem fraction
    '''    

    walkers = run_mcmc(data, sample, nwalkers, nsteps, phi_guess)
    traces = np.concatenate([sampler.chain[:,burn_in_time:,:], sampler.lnprobability[:,burn_in_time:,np.newaxis]],axis=2).reshape(-1, ndim+1).T
    return pd.DataFrame({key: traces[val] for val, key in enumerate(['vol_frac','l0','l1','lnprob'])})

def run_mcmc(data, sample, nwalkers, nsteps, phi_guess):
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
    '''    
    # set expected values to initialize walkers
    expected_vals = np.array([phi_guess,.5,0])
    ndim = len(expected_vals)
    
    # set walkers in a distribution with width .05
    starting_positions = [expected_vals + 0.05*np.random.randn(ndim) for i in range(nwalkers)]
    
    # figure out how many threads to use
    nthreads = np.min([nwalkers, mp.cpu_count()])

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=[data, sample, phi_guess], threads=nthreads)
    sampler.run_mcmc(starting_positions, nsteps)
    return sampler


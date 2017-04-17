import multiprocessing as mp
import numpy as np
import pandas as pd
from .main import rescale

def calc_prior(theta, phi_guess):
    vol_frac, l0, l1 = theta

    if l0 < 0 or l0 > 1 or l0+l1 <0 or l0+l1 > 1:
        # Losses are not in range [0,1] for some wavelength
        return -np.inf 

    if vol_frac < .35 or vol_frac > .73:
        # Outside range of validity of multiple scattering model
        return -np.inf
    
    var = .0025 # based on expected normal range
    return np.exp(-(vol_frac-phi_guess)**2/var)

def calc_likelihood(data, theory, l0, l1):
    """
    returns likelihood of obtaining an experimental dataset from a given spectrum
    
    Parameters:
        data: Experimental measurements (Spectrum object)
        theory: MC calculation results (Spectrum object)
        l0: constant loss parameter (float)
        l1: effect of wavelength on losses (float)
    """
    loss = l0 + l1*rescale(data.wavelength)
    residual = data.reflectance - (1-loss)*theory.reflectance
    var_eff = data.sigma_r**2 + theory.sigma_r**2
    chi_square = np.sum(residual**2/var_eff)
    prefactor = 1/np.prod(np.sqrt(2*np.pi*var_eff))
    return prefactor * np.exp(-chi_square/2)

def log_posterior(theta, data, sample, phi_guess):
    """
    calculates log-posterior of a set of parameters producing an observed reflectance spectrum

    Parameters:
        theta: set of parameters (list-like)
        data: contains observed reflectance and uncertainties (Spectrum object)
        sample: contains information about the sample that produced data (Sample object)
        phi_guess: expected volume fraction of the sample
    """
    vol_frac, l0, l1 = theta
    if not sample.wavelength == data.wavelength:
        raise ValueError("Sample and data must share the same set of wavelengths")

    theory_spectrum = do_some_monte_carlo_calculation(vol_frac, sample)
                        # returns a spectrum_object & assumes some uncertainty

    likelihood = calc_likelihood(data_spectrum, theory_spectrum, l0, l1)
    return np.log(likelihood) + np.log(calc_prior(theta, phi_guess))

def sample_parameters(data, sample, nwalkers=50, nsteps=500, burn_in_time=0, phi_guess = 0.55):
    """
    performs MCMC calculation and outputs DataFrame of parameters and log-probability
    Parameters:
        data: contains observed reflectance and uncertainties (Spectrum object)
        sample: contains information about the sample that produced data (Sample object)
        nwalkers: number of parallelized walkers to use in MCMC computation
        nsteps: total number of steps taken by each walker
        burn_in_time: number of inital steps to be removed from the resturned samples
        phi_guess: expected volume fraction of the sample (defaults to 0.55),
            chosen as a reasonable mid-point between dilute suspension and close-packed crystal
    """

    # set expected values to initialize walkers
    expected_vals = np.array([phi_guess,.5,0])
    ndim = len(expected_vals)
    
    # set walkers in a distribution with width .05
    starting_positions = [expected_vals + 0.05*np.random.randn(ndim) for i in range(nwalkers)]
    
    # figure out how many threads to use
    nthreads = np.min(nwalkers, mp.cpu_count())

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=[data, sample, phi_guess], threads=nthreads)
    sampler.run_mcmc(starting_positions, nsteps)
    traces = np.concatenate([sampler.chain[:,burn_in_time:,:], sampler.lnprobability[:,burn_in_time:,np.newaxis]],axis=2).reshape(-1, ndim+1).T
    return pd.DataFrame({key: traces[val] for val, key in enumerate(['vol_frac','l0','l1','lnprob'])})

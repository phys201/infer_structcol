'''
This file contains the generative model to calculate posterior probabilities.
'''

import multiprocessing as mp
import numpy as np
import pandas as pd
import emcee
from .main import Spectrum, rescale, check_wavelength
from .run_structcol import calc_reflectance

# define limits of validity for the MC scattering model
min_phi = 0.35
max_phi = 0.73

def calc_model_spect(sample, theta, seed=None):
    ''''
    Calculates a corrected theoretical spectrom from a set of parameters.
    
    Parameters
    -------
    sample: Sample object
        information about the sample that produced data_spectrum
    theta: 3-tuple 
        set of inference parameter values - volume fraction, baseline loss, wavelength dependent loss
    seed: int (optional)
        if specified, passes the seed through to the MC multiple scatterin calculation
    '''

    phi, l0, l1 = theta
    loss = l0 + l1*rescale(sample.wavelength)
    theory_spectrum = calc_reflectance(phi, sample, seed=seed)
    theory_spectrum['reflectance'] *= (1-loss)
    theory_spectrum['sigma_r'] *= (1-loss)
    return theory_spectrum

def calc_resid_spect(spect1, spect2):
    '''
    Calculates the difference between two spectra and convolves their uncertainty.
    
    Parameters
    -------
    spect1: Spectrum object
        one of two spectra to be compared
    spect2: Spectrum object
        second of two spectra to be compared

    Returns
    -------
    Spectrum object: contains residuals and uncertainties at each wavelength
    '''
    residual = spect1.reflectance - spect2.reflectance
    sigma_eff = np.sqrt(spect1.sigma_r**2 + spect2.sigma_r**2)
    return Spectrum(check_wavelength(spect1, spect2), residual, sigma_eff)

def calc_log_prior(theta):
    '''
    Calculats log of prior probability of obtaining theta.
    
    Parameters
    -------
    theta: 3-tuple 
        set of inference parameter values - volume fraction, baseline loss, wavelength dependent loss
    '''
    vol_frac, l0, l1 = theta

    if l0 < 0 or l0 > 1 or l0+l1 <0 or l0+l1 > 1:
        # Losses are not in range [0,1] for some wavelength
        return -np.inf 

    if not min_phi < vol_frac < max_phi:
        # Outside range of validity of multiple scattering model
        return -np.inf

    return 0

def calc_likelihood(spect1, spect2):
    '''
    Returns likelihood of obtaining an experimental dataset from a given theoretical spectrum

    Parameters
    -------
    spect1: Spectrum object
        experimental dataset
    spect2: Spectrum object
        calculated dataset
    '''
    resid_spect = calc_resid_spect(spect1, spect2)
    chi_square = np.sum(resid_spect.reflectance**2/resid_spect.sigma_r**2)
    prefactor = 1/np.prod(resid_spect.sigma_r * np.sqrt(2*np.pi))
    return prefactor * np.exp(-chi_square/2)

def log_posterior(theta, data_spectrum, sample, seed=None):
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
    seed: int (optional)
        if specified, passes the seed through to the MC multiple scatterin calculation
    '''    
    check_wavelength(data_spectrum, sample) # not used for anything, but we need to run the check.
    log_prior = calc_log_prior(theta)
    if log_prior == -np.inf:
        # don't bother running MC
        return -1e100
        return log_prior

    theory_spectrum = calc_model_spect(sample, theta, seed)
    likelihood = calc_likelihood(data_spectrum, theory_spectrum)
    return np.log(likelihood) + log_prior

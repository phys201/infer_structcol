'''
This file contains the generative model to calculate posterior probabilities.
'''

import multiprocessing as mp
import numpy as np
import pandas as pd
import emcee
from .main import Spectrum, rescale, check_wavelength
from .run_structcol import calc_refl_trans, calc_sigma

minus_inf = -1e100 # required since emcee throws errors if we actually pass in -inf

def calc_model_spect(sample, theta, sigma, ntrajectories, nevents, seed=None):
    ''''
    Calculates a corrected theoretical spectrum from a set of parameters.
    
    Parameters
    -------
    sample: Sample object
        information about the sample that produced data_spectrum
    theta: 5- or 7-tuple 
        set of inference parameter values - volume fraction, particle radius, 
        thickness, reflection baseline loss, reflection wavelength dependent loss, 
        transmission baseline loss, transmission wavelength dependent loss
    sigma: 2-tuple
        uncertainties (taken to be 1 standard deviation) of the multiple scattering
        calculations (reflectance sigma, transmittance sigma).
    ntrajectories: int
        number of trajectories for the multiple scattering calculations
    nevents: int
        number of scattering events for the multiple scattering calculations
    seed: int (optional)
        if specified, passes the seed through to the MC multiple scattering 
        calculation
    '''
    if len(theta) == 7:
        phi, radius, thickness, l0_r, l1_r, l0_t, l1_t = theta
        loss_r = l0_r + l1_r*rescale(sample.wavelength)
        loss_t = l0_t + l1_t*rescale(sample.wavelength)
    if len(theta) == 5:
        phi, radius, thickness, l0, l1 = theta 
        loss_r = l0 + l1*rescale(sample.wavelength)
        loss_t = loss_r
    
    # Calculate the reflectance and transmittance spectra with the multiple
    # scattering model
    sigma_r, sigma_t = sigma
    refl, trans = calc_refl_trans(phi, radius, thickness, sample, ntrajectories, nevents, seed=seed)
    
    # Make a spectrum object out of the loss-corrected spectra and their standard deviations 
    theory_spectrum = Spectrum(sample.wavelength, reflectance = refl, 
                               transmittance = trans, sigma_r = sigma_r, sigma_t = sigma_t)
    theory_spectrum['reflectance'] *= (1-loss_r)
    theory_spectrum['sigma_r'] *= (1-loss_r)
    theory_spectrum['transmittance'] *= (1-loss_t)
    theory_spectrum['sigma_t'] *= (1-loss_t)
    
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
    nan_array = np.full(spect1.wavelength.shape, np.nan)
    residual_r = nan_array
    residual_t = nan_array
    sigma_eff_r = nan_array
    sigma_eff_t = nan_array

    if 'reflectance' in spect1.keys() and 'reflectance' in spect2.keys():
        residual_r = spect1.reflectance - spect2.reflectance
        sigma_eff_r = np.sqrt(spect1.sigma_r**2 + spect2.sigma_r**2)

    if 'transmittance' in spect1.keys() and 'transmittance' in spect2.keys():
        residual_t = spect1.transmittance - spect2.transmittance
        sigma_eff_t = np.sqrt(spect1.sigma_t**2 + spect2.sigma_t**2)

    return Spectrum(check_wavelength(spect1, spect2), reflectance = residual_r, 
                    sigma_r = sigma_eff_r, transmittance = residual_t, 
                    sigma_t = sigma_eff_t)

def calc_log_prior(theta, theta_range):
    '''
    Calculates log of prior probability of obtaining theta.
    
    Parameters
    -------
    theta: 5-, 7-tuple 
        set of inference parameter values - volume fraction, particle radius, 
        thickness, baseline loss, wavelength dependent loss
    theta_range: dictionary
        best guess of the expected ranges of the parameter values 
        (min_phi, max_phi, min_radius, max_radius, min_thickness, max_thickness, 
        min_l0_r, max_l0_r, min_l1_r, max_l1_r, min_l0_t, max_l0_t, min_l1_t, max_l1_t)) 
    
    '''
    if len(theta) == 7:
        vol_frac, radius, thickness, l0_r, l1_r, l0_t, l1_t = theta
        if l0_r < 0 or l0_r > 1 or l0_r+l1_r <0 or l0_r+l1_r > 1:
            # Losses are not in range [0,1] for some wavelength
            return -np.inf 
        if l0_t < 0 or l0_t > 1 or l0_t+l1_t <0 or l0_t+l1_t > 1:
            # Losses are not in range [0,1] for some wavelength
            return -np.inf 
    if len(theta) == 5:
        vol_frac, radius, thickness, l0, l1 = theta
        if l0 < 0 or l0 > 1 or l0+l1 <0 or l0+l1 > 1:
            # Losses are not in range [0,1] for some wavelength
            return -np.inf 

    if not theta_range['min_phi'] < vol_frac < theta_range['max_phi']:
        # Outside range of prior values
        return -np.inf

    if not theta_range['min_radius'] < radius < theta_range['max_radius']:
        # Outside range of prior values
        return -np.inf
    
    if not theta_range['min_thickness'] < thickness < theta_range['max_thickness']:
        # Outside range of prior values
        return -np.inf
    
    return 0

def calc_likelihood(spect1, spect2):
    '''
    Returns likelihood of obtaining an experimental dataset from a given 
    theoretical spectrum

    Parameters
    ----------
    spect1: Spectrum object
        experimental dataset
    spect2: Spectrum object
        calculated dataset
    '''
    resid_spect = calc_resid_spect(spect1, spect2)
    chi_square = 0.
    prefactor = 1.

    if 'reflectance' in spect1.keys() and 'reflectance' in spect2.keys():
        chi_square += np.sum(resid_spect.reflectance**2/resid_spect.sigma_r**2)
        prefactor *= 1/np.prod(resid_spect.sigma_r * np.sqrt(2*np.pi))

    if 'transmittance' in spect1.keys() and 'transmittance' in spect2.keys():
        chi_square += np.sum(resid_spect.transmittance**2/resid_spect.sigma_t**2)
        prefactor *= 1/np.prod(resid_spect.sigma_t * np.sqrt(2*np.pi))
        
    return prefactor * np.exp(-chi_square/2)

def log_posterior(theta, data_spectrum, sample, theta_range, sigma, ntrajectories, nevents, seed=None):
    '''
    Calculates log-posterior of a set of parameters producing an observed 
    reflectance spectrum
    
    Parameters
    ----------
    theta: 5- or 7-tuple 
        set of inference parameter values - volume fraction, particle radius, 
        thickness, baseline loss, wavelength dependent loss
    data_spectrum: Spectrum object
        experimental dataset
    sample: Sample object
        information about the sample that produced data_spectrum
    theta_range: dictionary
        best guess of the expected ranges of the parameter values 
        (min_phi, max_phi, min_radius, max_radius, min_thickness, max_thickness, 
        min_l0_r, max_l0_r, min_l1_r, max_l1_r, min_l0_t, max_l0_t, min_l1_t, max_l1_t)) 
    sigma: 2-tuple
        uncertainties (taken to be 1 standard deviation) of the multiple scattering
        calculations (reflectance sigma, transmittance sigma).
    ntrajectories: int
        number of trajectories for the multiple scattering calculations
    nevents: int
        number of scattering events for the multiple scattering calculations
    seed: int (optional)
        if specified, passes the seed through to the MC multiple scattering 
        calculation
    '''    
    check_wavelength(data_spectrum, sample) # not used for anything, but we need to run the check.
    log_prior = calc_log_prior(theta, theta_range)
    if log_prior == -np.inf:
        # don't bother running MC
        return minus_inf

    theory_spectrum = calc_model_spect(sample, theta, sigma, ntrajectories, nevents, seed)
    likelihood = calc_likelihood(data_spectrum, theory_spectrum)

    if likelihood == 0:
        # don't bother running MC
        return minus_inf
    
    return np.log(likelihood) + log_prior

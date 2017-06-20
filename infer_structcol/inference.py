'''
This file contains the inference framework to estimate sample properties.
'''

import multiprocessing as mp
import numpy as np
import pandas as pd
import emcee
import lmfit
from .model import (calc_model_spect, calc_resid_spect, log_posterior)

# define limits of validity for the MC scattering model
theta_range_default = {'min_phi':0.1, 'max_phi':0.74, 'min_radius':10., 
                       'max_radius': 1000., 'min_thickness':1., 'max_thickness':1000., 
                       'min_l0_r':0, 'max_l0_r':1, 'min_l1_r':-1, 'max_l1_r':1,
                       'min_l0_t':0, 'max_l0_t':1, 'min_l1_t':-1, 'max_l1_t':1}  # radius in nm, thickness in um

# define initial guesses for theta in case the user doesn't give an initial guess
theta_guess_default = {'phi':0.5, 'radius':120, 'thickness':100, 'l0_r':0.02, 
                       'l1_r':0, 'l0_t':0.02, 'l1_t':0}  # radius in nm, thickness in um

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
    theta_guess: dictionary 
        best guess of the expected parameter values (phi, radius, thickness, l0_r, l1_r, l0_t, l1_t)
    theta_range: dictionary
        expected ranges of the parameter values 
        (min_phi, max_phi, min_radius, max_radius, min_thickness, max_thickness, 
        min_l0_r, max_l0_r, min_l1_r, max_l1_r, min_l0_t, max_l0_t, min_l1_t, max_l1_t) 
    
    Returns
    -------
    theta: 5 or 7 -tuple
        best fit (phi, radius, thickness, l0, l1) or (phi, radius, thickness, 
        l0_r, l1_r, l0_t, l1_t) as floats
    '''
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
    for key in theta_guess:
        fit_params[key] = lmfit.Parameter(value=theta_guess[key], min=theta_range['min_'+key], max=theta_range['max_'+key])
    fit_params = lmfit.minimize(resid, fit_params).params

    return tuple(fit_params.valuesdict().values())


def run_mcmc(data, sample, nwalkers, nsteps, theta_guess = theta_guess_default, theta_range = theta_range_default, seed=None):
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
    theta_guess: dictionary (optional)
        user's best guess of the expected parameter values (phi, radius, thickness)
    theta_range: dictionary (optional)
        user's best guess of the expected ranges of the parameter values 
        (min_phi, max_phi, min_radius, max_radius, min_thickness, max_thickness) 
    seed: int (optional)
        sets the seed for all MC scattering trajectory chains. 
        DOES NOT set the seed for MCMC walkers.
    '''    

    # set expected ranges of values to initialize walkers. If the user inputs an incorrect key, then raise an Error
    for key in theta_range:
        if key not in theta_range_default:
            raise KeyError('Parameter {0} in theta_range is not defined correctly'.format(str(key)))
    theta_range_default.update(theta_range) 
    theta_range = theta_range_default

    # set initial guesses to initialize walkers. If the user inputs an incorrect key, then raise an Error
    for key in theta_guess:
        if key not in theta_guess_default:
            raise KeyError('Parameter {0} in theta_guess is not defined correctly'.format(str(key)))
    theta_guess_default.update(theta_guess) 
    theta_guess = theta_guess_default

    theta = find_max_like(data, sample, theta_guess, theta_range, seed)
    ndim = len(theta)
      
    # set walkers in a distribution with width .05
    vf = np.clip(theta[0]*np.ones(nwalkers) + 0.01*np.random.randn(nwalkers), 
                 theta_range['min_phi'], theta_range['max_phi'])
    radius = np.clip(theta[1]*np.ones(nwalkers) + 1*np.random.randn(nwalkers), 
                 theta_range['min_radius'], theta_range['max_radius'])
    thickness = np.clip(theta[2]*np.ones(nwalkers) + 1*np.random.randn(nwalkers), 
                 theta_range['min_thickness'], theta_range['max_thickness'])
    l0 = np.clip(theta[3]*np.ones(nwalkers) + 0.01*np.random.randn(nwalkers), 
                 theta_range['min_l0_r'], theta_range['max_l0_r'])
    # clip the sum of l0 and l1 and subtract l0 to ensure that the clipped 
    # l1 satisfies 0 < l0 + l1 < 1        
    l1 = np.clip((theta[3] + theta[4]) * np.ones(nwalkers) + 
                0.01*np.random.randn(nwalkers), theta_range['min_l1_r'], theta_range['max_l0_r']) - l0
    
    if ndim == 5:
        theta = np.vstack((vf,radius,thickness,l0,l1)).T.tolist()

    if ndim == 7:
        l0_1 = np.clip(theta[5] * np.ones(nwalkers) + 
                       0.01*np.random.randn(nwalkers), theta_range['min_l0_t'], theta_range['max_l0_t'])
        l1_1 = np.clip((theta[5] + theta[6]) * np.ones(nwalkers) + 
                       0.01*np.random.randn(nwalkers), theta_range['min_l1_t'], theta_range['max_l1_t']) - l0_1
        theta = np.vstack((vf,radius,thickness,l0,l1,l0_1,l1_1)).T.tolist()

    # figure out how many threads to use
    nthreads = np.min([nwalkers, mp.cpu_count()])

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, 
                                    args=[data, sample, theta_range, seed], threads=nthreads)
    sampler.run_mcmc(theta, nsteps)

    return sampler
    

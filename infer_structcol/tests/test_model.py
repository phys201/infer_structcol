'''
This file tests functions from model.py.
'''
import structcol as sc
from infer_structcol.model import *
from infer_structcol.main import Spectrum, Sample, find_close_indices
import numpy as np
from numpy.testing import assert_equal, assert_approx_equal
from pandas.util.testing import assert_frame_equal
from .test_inference import (ntrajectories, nevents, wavelength_sigma, sigma)

def test_calc_model_spect():
    wavelength = [500]
    sample = Sample(wavelength, 1.5, 1)
    theta = (0.5, 100, 200, 0, 0, 0, 0)
    
    wavelength_ind = find_close_indices(wavelength_sigma, sc.Quantity(wavelength,'nm'))
    sigma_test = sigma[np.array(wavelength_ind)]
    assert_frame_equal(calc_model_spect(sample, theta, (sigma_test, sigma_test), ntrajectories, nevents, 2), 
                       Spectrum(500, reflectance = 0.828595524325, sigma_r = 0.0193369922424, 
                       transmittance =0.171404475675, sigma_t = 0.0193369922424))

def test_calc_resid_spect():
    spect1=Spectrum(500, reflectance = 0.5, transmittance = 0.5, sigma_r = 0.1, 
                    sigma_t = 0.1)
    spect2=Spectrum(500, reflectance = 0.7, sigma_r = 0)
    expected_output = Spectrum(500, reflectance = 0.2, sigma_r = 0.1, 
                               transmittance = np.nan, sigma_t = np.nan)
    assert_frame_equal(calc_resid_spect(spect2, spect1), expected_output)

def test_log_prior():
    theta_range = {'min_phi':0.35, 'max_phi':0.74, 'min_radius':70, 'max_radius': 160, 
               'min_thickness':1, 'max_thickness':1000}

    # Test different conditions with only reflectance or transmittance 
    assert_approx_equal(calc_log_prior((0.5, 100, 100, 0, 1), theta_range), 0)
    assert_approx_equal(calc_log_prior((0.5, 100, 100, 1,-1), theta_range), 0)
    assert_equal(calc_log_prior((0.5, 100, 100, -0.5,1), theta_range), -np.inf)
    assert_equal(calc_log_prior((0.5, 100, 100, 0,1.5), theta_range), -np.inf)
    assert_equal(calc_log_prior((1.0, 100, 100, 1, 1), theta_range), -np.inf)
    assert_equal(calc_log_prior((0.5, 10, 100, 0, 1), theta_range), -np.inf)
    assert_equal(calc_log_prior((0.5, 300, 100, 0, 1), theta_range), -np.inf)
    assert_equal(calc_log_prior((0.5, 100, 100, 0, 1), theta_range), calc_log_prior((0.6, 100, 100, 0, 1), theta_range))
    assert_equal(calc_log_prior((0.5, 100, 1001, 0.5, 0.1), theta_range), -np.inf)
    
    # Tests for when there is both reflectance and transmittance 
    assert_approx_equal(calc_log_prior((0.5, 100, 100, 0, 0, 0, 1), theta_range), 0)
    assert_approx_equal(calc_log_prior((0.5, 100, 100, 0, 0, 1, -1), theta_range), 0)
    assert_equal(calc_log_prior((0.5, 100, 100, -0.5, 1, -0.5, 1), theta_range), -np.inf)
    assert_equal(calc_log_prior((0.5, 100, 100, 0, 1.5, 0, 1.5), theta_range), -np.inf)
    assert_equal(calc_log_prior((1.0, 100, 100, 1, 1, 1, 1), theta_range), -np.inf)
    assert_approx_equal(calc_log_prior((0.5, 10, 100, 0, 0, 0, 1), theta_range), -np.inf)
    assert_approx_equal(calc_log_prior((0.5, 300, 100, 0, 0, 1, -1), theta_range), -np.inf)
    assert_equal(calc_log_prior((0.5, 100, 100, 0, 1, 0, 1), theta_range), calc_log_prior((0.6, 100, 100, 0, 1, 0, 1), theta_range))
    assert_equal(calc_log_prior((0.5, 100, 1001, 0.5, 0.1, 0.1, 0.1), theta_range), -np.inf)
    
def test_likelihood():
    spect1=Spectrum(500, reflectance = 0.5, transmittance = 0.5, sigma_r = 0.1, sigma_t = 0.)
    spect2=Spectrum(500, reflectance = 0.7, transmittance = 0.3, sigma_r = 0., sigma_t = 0.1)
    expected_output = (1 / np.sqrt(2*np.pi*0.01) * np.exp(-2))**2
    assert_approx_equal(calc_likelihood(spect1, spect2), expected_output)

def test_log_posterior():
    wavelength = [500]
    spectrum=Spectrum(wavelength, reflectance = 0.5, sigma_r = 0.1)
    sample = Sample(wavelength, 1.5, 1)
    theta_range = {'min_phi':0.35, 'max_phi':0.74, 'min_radius':70, 'max_radius': 201, 
               'min_thickness':1, 'max_thickness':1000}
    wavelength_ind = find_close_indices(wavelength_sigma, sc.Quantity(wavelength,'nm'))
    sigma_test = sigma[np.array(wavelength_ind)]
    
    # When parameters are within prior range
    theta1 = (0.5, 200, 200, 0, 0)
    post1 = log_posterior(theta1, spectrum, sample, theta_range=theta_range, 
                          sigma=(sigma_test, sigma_test), ntrajectories=ntrajectories, nevents=nevents, seed=2)
    assert_approx_equal(post1, -6.0413752765269875)
    
    # When parameters are not within prior range
    theta2 = (0.3, 200, 200, 0, 0)
    post2 = log_posterior(theta2, spectrum, sample, theta_range=theta_range, 
                          sigma=(sigma_test, sigma_test), ntrajectories=ntrajectories, nevents=nevents, seed=2)
    assert_approx_equal(post2, -1e100)
    

'''
This file tests functions from model.py.
'''

from infer_structcol.model import *
from infer_structcol.main import Spectrum, Sample
import numpy as np
from numpy.testing import assert_equal, assert_approx_equal
from pandas.util.testing import assert_frame_equal
import warnings


def test_calc_model_spect():
    sample = Sample(500, 100, 200, 1.5, 1)
    theta = (0.5, 0, 0, 0, 0)
    assert_frame_equal(calc_model_spect(sample, theta, 2), Spectrum(500, 
                       reflectance = 0.813006364656, sigma_r = 0.025358376581, 
                       transmittance =0.186993635344, sigma_t = 0.025358376581))

def test_calc_resid_spect():
    spect1=Spectrum(500, reflectance = 0.5, transmittance = 0.5, sigma_r = 0.1, 
                    sigma_t = 0.1)
    spect2=Spectrum(500, reflectance = 0.7, sigma_r = 0)
    expected_output = Spectrum(500, reflectance = 0.2, sigma_r = 0.1, 
                               transmittance = np.nan, sigma_t = np.nan)
    assert_frame_equal(calc_resid_spect(spect2, spect1), expected_output)

def test_log_prior():
    # Test different conditions with only reflectance or transmittance 
    assert_approx_equal(calc_log_prior((0.5, 0, 1)), 0)
    assert_approx_equal(calc_log_prior((0.5, 1,-1)), 0)
    assert_equal(calc_log_prior((0.5,-0.5,1)), -np.inf)
    assert_equal(calc_log_prior((0.5, 0,1.5)), -np.inf)
    assert_equal(calc_log_prior((1.0, 1, 1)), -np.inf)
    assert_equal(calc_log_prior((0.5, 0, 1)), calc_log_prior((0.6, 0, 1)))

    # Tests for when there is both reflectance and transmittance 
    assert_approx_equal(calc_log_prior((0.5, 0, 0, 0, 1)), 0)
    assert_approx_equal(calc_log_prior((0.5, 0, 0, 1, -1)), 0)
    assert_equal(calc_log_prior((0.5,-0.5, 1, -0.5, 1)), -np.inf)
    assert_equal(calc_log_prior((0.5, 0, 1.5, 0, 1.5)), -np.inf)
    assert_equal(calc_log_prior((1.0, 1, 1, 1, 1)), -np.inf)
    assert_equal(calc_log_prior((0.5, 0, 1, 0, 1)), calc_log_prior((0.6, 0, 1, 0, 1)))
    
def test_likelihood():
    spect1=Spectrum(500, reflectance = 0.5, transmittance = 0.5, sigma_r = 0.1, sigma_t = 0.)
    spect2=Spectrum(500, reflectance = 0.7, transmittance = 0.3, sigma_r = 0., sigma_t = 0.1)
    expected_output = (1 / np.sqrt(2*np.pi*0.01) * np.exp(-2))**2
    assert_approx_equal(calc_likelihood(spect1, spect2), expected_output)

def test_log_posterior():
    warnings.simplefilter('ignore', UserWarning)
    spectrum=Spectrum(500, reflectance = 0.5, sigma_r = 0.1)
    sample = Sample(500, 200, 200, 1.5, 1)

    # When parameters are within prior range
    theta1 = (0.5, 0, 0)
    post1 = log_posterior(theta1, spectrum, sample, seed=2)
    assert_approx_equal(post1, -6.329515523909971)
    
    # When parameters are not within prior range
    theta2 = (0.3, 0, 0)
    post2 = log_posterior(theta2, spectrum, sample, seed=2)
    assert_approx_equal(post2, -1e100)
    

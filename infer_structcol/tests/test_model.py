from infer_structcol.model import *
from infer_structcol.main import Spectrum, Sample
import numpy as np
from numpy.testing import assert_equal, assert_approx_equal
from pandas.util.testing import assert_frame_equal

def test_calc_model_spect():
    sample = Sample(500, 200, 200, 1.5, 1)
    theta = (0.5, 0, 0)
    assert_frame_equal(calc_model_spect(sample, theta, 2), Spectrum(500, reflectance = 0.507608289932, sigma_r = 0.0261746352879))

def test_calc_resid_spect():
    spect1=Spectrum(500, reflectance = 0.5, sigma_r = 0.1)
    spect2=Spectrum(500, reflectance = 0.7, sigma_r = 0)
    expected_output = Spectrum(500, reflectance = 0.2, sigma_r = 0.1)
    assert_frame_equal(calc_resid_spect(spect2, spect1), expected_output)

def test_prior():
    pars = (0.5, 0, 1)
    phi_guess = 0.5
    assert_approx_equal(calc_log_prior((0.5, 0, 1)), 0)
    assert_approx_equal(calc_log_prior((0.5, 1,-1)), 0)
    assert_equal(calc_log_prior((0.5,-0.5,1)), -np.inf)
    assert_equal(calc_log_prior((0.5, 0,1.5)), -np.inf)
    assert_equal(calc_log_prior((1.0, 1, 1)), -np.inf)
    assert_equal(calc_log_prior((0.5, 0, 1)), calc_log_prior((0.6, 0, 1)))

def test_likelihood():
    spect1=Spectrum(500, reflectance = 0.5, sigma_r = 0.1)
    spect2=Spectrum(500, reflectance = 0.7, sigma_r = 0)
    expected_output1 = 1/np.sqrt(2*np.pi*0.01)*np.exp(-2)
    assert_approx_equal(calc_likelihood(spect1, spect2), expected_output1)

def test_log_posterior():
    spectrum=Spectrum(500, reflectance = 0.5, sigma_r = 0.1)
    theta = (0.5, 0, 0)
    sample = Sample(500, 200, 200, 1.5, 1)
    post = log_posterior(theta, spectrum, sample, seed=2)
    assert_approx_equal(post, 1.3478047169617922)
    

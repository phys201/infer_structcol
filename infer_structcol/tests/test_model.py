from infer_structcol.model import calc_log_prior, calc_likelihood, log_posterior 
from infer_structcol.main import Spectrum, Sample
import numpy as np
from numpy.testing import assert_equal, assert_approx_equal

def test_prior():
    pars = (0.5, 0, 1)
    phi_guess = 0.5
    assert_approx_equal(calc_log_prior((0.5, 0, 1), 0.5), 0)
    assert_approx_equal(calc_log_prior((0.5, 1,-1), 0.5), 0)
    assert_equal(calc_log_prior((0.5,-0.5,1),0.5), -np.inf)
    assert_equal(calc_log_prior((0.5, 0,1.5),0.5), -np.inf)
    assert_equal(calc_log_prior((1.0, 1, 1), 0.5), -np.inf)
    assert_equal(calc_log_prior((0.5, 0, 1), 0.6), calc_log_prior((0.6, 0, 1), 0.5))

def test_likelihood():
    spect1=Spectrum(500, 0.5, 0.1)
    spect2=Spectrum(500, 0.7, 0)
    expected_output1 = 1/np.sqrt(2*np.pi*0.01)*np.exp(-2)
    assert_approx_equal(calc_likelihood(spect1, spect2, 0, 0), expected_output1)

    expected_output2 = 1/np.sqrt(2*np.pi*(0.01))
    assert_approx_equal(calc_likelihood(spect1, spect2, 2/7, 0), expected_output2)

def test_log_posterior():
    spectrum = Spectrum(500, 0.5, 0.1)
    theta = (0.5, 0, 0)
    sample = Sample(500, 200, 200, 1.5, 1)
    post = log_posterior(theta, spectrum, sample, 0.5, seed=2)
    assert_approx_equal(post, 1.3478047169617922)
    

from infer_structcol.model import calc_prior, calc_likelihood, log_posterior, sample_parameters
from infer_structcol.main import Spectrum
import numpy as np
from numpy.testing import assert_equal, assert_approx_equal

def test_prior():
    pars = (0.5, 0, 1)
    phi_guess = 0.5
    assert_equal(calc_prior((0.5, 0, 1), 0.5), 1)
    assert_equal(calc_prior((0.5, 1,-1), 0.5), 1)
    assert_equal(calc_prior((0.5,-0.5,1),0.5), -np.inf)
    assert_equal(calc_prior((0.5, 0,1.5),0.5), -np.inf)
    assert_equal(calc_prior((1.0, 1, 1), 0.5), -np.inf)
    assert_equal(calc_prior((0.5, 0, 1), 0.6), calc_prior((0.6, 0, 1), 0.5))

def test_likelihood():
    spect1=Spectrum(500, 0.5, 0.1)
    spect2=Spectrum(500, 0.7, 0.1)
    expected_output = 1/np.sqrt(2*np.pi*0.02)*np.exp(-1)
    assert_approx_equal(calc_likelihood(spect1, spect2, 0, 0), expected_output)

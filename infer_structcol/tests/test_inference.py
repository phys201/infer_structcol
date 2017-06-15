'''
This file tests functions from inference.py.
'''

from infer_structcol.inference import find_max_like, run_mcmc
from infer_structcol.main import Sample, Spectrum
from infer_structcol.model import calc_model_spect
from numpy.testing import assert_equal, assert_almost_equal
import numpy as np

def test_find_max_like():
    sample = Sample([450, 500, 550, 600], particle_index=1.59, matrix_index=1)
    theta = (0.55, 120, 120, 0.02, 0, 0.02, 0) #these are the default starting values for lmfit calculation
    spect = calc_model_spect(sample, theta, seed=2)
    
    theta_range = np.array([[0.34,0.74], [70, 160], [1, 1000]])
    max_like_vals = find_max_like(spect, sample, theta_guess=None, theta_range=theta_range, seed=2)
    assert_almost_equal(max_like_vals, theta)

def test_run_mcmc():
    spectrum = Spectrum([500, 550, 600, 650], reflectance = [0.5, 0.5, 0.5, 0.5], transmittance = [0.5, 0.5, 0.5, 0.5], 
                        sigma_r = [0.1, 0.1, 0.1, 0.1], sigma_t = [0.1, 0.1, 0.1, 0.1])
    sample = Sample([500, 550, 600, 650], 1.5, 1)
    theta = (0.5, 150, 120, 0, 0, 0, 0)
    # Test that run_mcmc runs correctly
    run_mcmc(spectrum, sample, nwalkers=14, nsteps=1, theta_guess=theta, theta_range=None, seed=3)

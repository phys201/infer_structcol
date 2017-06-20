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
    
    theta_range = {'min_phi':0.34, 'max_phi':0.74, 'min_radius':70, 'max_radius': 160, 
               'min_thickness':1, 'max_thickness':1000, 'min_l0_r':0, 'max_l0_r':1, 
               'min_l1_r':-1, 'max_l1_r':1, 'min_l0_t':0, 'max_l0_t':1, 'min_l1_t':-1, 'max_l1_t':1}
    theta_guess = {'phi':0.55, 'radius':120, 'thickness':120, 'l0_r':0.02, 
                       'l1_r':0, 'l0_t':0.02, 'l1_t':0} 
    max_like_vals = find_max_like(spect, sample, theta_guess=theta_guess, theta_range=theta_range, seed=2)
    assert_almost_equal(max_like_vals, theta)

def test_run_mcmc():
    spectrum = Spectrum([500, 550, 600, 650], reflectance = [0.5, 0.5, 0.5, 0.5], transmittance = [0.5, 0.5, 0.5, 0.5], 
                        sigma_r = [0.1, 0.1, 0.1, 0.1], sigma_t = [0.1, 0.1, 0.1, 0.1])
    sample = Sample([500, 550, 600, 650], 1.5, 1)
    theta_guess = {'phi':0.5, 'radius':150, 'thickness':120, 'l0_r':0, 
                       'l1_r':0, 'l0_t':0, 'l1_t':0} 
    theta_range = {'min_phi':0.35, 'max_phi':0.73, 'min_radius':70, 'max_radius': 201, 
               'min_thickness':1, 'max_thickness':1000, 'min_l0_r':0, 'max_l0_r':1, 
               'min_l1_r':-1, 'max_l1_r':1, 'min_l0_t':0, 'max_l0_t':1, 'min_l1_t':-1, 'max_l1_t':1}
    # Test that run_mcmc runs correctly
    run_mcmc(spectrum, sample, nwalkers=14, nsteps=1, theta_guess=theta_guess, theta_range=theta_range, seed=3)

'''
This file tests functions from inference.py.
'''
import structcol as sc
from infer_structcol.inference import find_max_like, run_mcmc
from infer_structcol.main import Sample, Spectrum, find_close_indices
from infer_structcol.model import calc_model_spect
from numpy.testing import assert_equal, assert_almost_equal
import numpy as np

ntrajectories = 600
nevents = 200
wavelength_sigma = sc.Quantity(np.linspace(400,1000,61),'nm')
sigma = np.array([1.860651421552072735e-02, 1.753980839818162357e-02, 1.839622738704549398e-02, 
                  1.596763386664768955e-02, 1.894484659740078986e-02, 1.722962247665738716e-02, 
                  1.555134197251030123e-02, 1.763293909648367200e-02, 2.027257609441594777e-02, 
                  1.850550125238413501e-02, 1.933699224240205058e-02, 1.873148138270526453e-02, 
                  1.908441182529240290e-02, 1.756355142274622708e-02, 1.590192651066632198e-02, 
                  1.596104976169695697e-02, 2.024553310180053287e-02, 1.955488448380025140e-02, 
                  1.882008022078682577e-02, 1.796507064336797313e-02, 2.004778422542081301e-02, 
                  1.811040666898488388e-02, 1.805909831464867776e-02, 1.810327867013098932e-02, 
                  1.516823124817042248e-02, 1.514314740128578328e-02, 1.696441336804245872e-02, 
                  1.677168419886158890e-02, 1.132382672347467811e-02, 1.224676407793331805e-02, 
                  1.117690246951372202e-02, 1.241312684961146107e-02, 1.326040920813134627e-02, 
                  1.367716094293736952e-02, 1.206014700075800326e-02, 1.250865278649789837e-02, 
                  1.060414384515132730e-02, 1.036736066347118505e-02, 9.217086600787521497e-03, 
                  1.110553603581512436e-02, 1.045612311627215976e-02, 9.731754980500811197e-03, 
                  9.142893085166936898e-03, 9.140217170604352653e-03, 8.149317528863142188e-03, 
                  7.833831850636231026e-03, 8.968239058921748802e-03, 7.848222412457096439e-03, 
                  7.609828884008617081e-03, 7.541218020424243600e-03, 7.964287577656070996e-03, 
                  7.665492573392837863e-03, 7.555737277343346943e-03, 7.034118091273018451e-03, 
                  6.271285385383303969e-03, 7.198737679024127048e-03, 5.980837995132812224e-03, 
                  6.166925243497538289e-03, 6.148309644049101616e-03, 6.087239500545048483e-03, 
                  6.549083556399931151e-03])

def test_find_max_like():
    wavelength = [450, 500, 550, 600]
    wavelength_ind = find_close_indices(wavelength_sigma, sc.Quantity(wavelength,'nm'))
    sigma_test = sigma[np.array(wavelength_ind)]
    
    sample = Sample(wavelength, particle_index=1.59, matrix_index=1)
    theta = (0.55, 120, 120, 0.02, 0, 0.02, 0) #these are the default starting values for lmfit calculation
    spect = calc_model_spect(sample, theta, (sigma_test, sigma_test), ntrajectories, nevents, seed=2)
    
    theta_range = {'min_phi':0.34, 'max_phi':0.74, 'min_radius':70, 'max_radius': 160, 
               'min_thickness':1, 'max_thickness':1000, 'min_l0_r':0, 'max_l0_r':1, 
               'min_l1_r':-1, 'max_l1_r':1, 'min_l0_t':0, 'max_l0_t':1, 'min_l1_t':-1, 'max_l1_t':1}
    theta_guess = {'phi':0.55, 'radius':120, 'thickness':120, 'l0_r':0.02, 
                       'l1_r':0, 'l0_t':0.02, 'l1_t':0} 
    
    max_like_vals = find_max_like(spect, sample, theta_guess=theta_guess, 
                                  theta_range=theta_range, sigma=(sigma_test, sigma_test), 
                                  ntrajectories=ntrajectories, nevents=nevents, seed=2)
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

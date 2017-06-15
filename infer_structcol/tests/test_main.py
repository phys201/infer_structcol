'''
This file tests functions from main.py.
'''

from infer_structcol.main import (Spectrum, Sample, extend_array, rescale, 
find_close_indices, check_wavelength, convert_dtype)
import numpy as np
from numpy.testing import assert_equal, assert_raises

def test_Spectrum():
    # Test if Spectrum object is created correctly when there is reflectance 
    # and transmittance
    spect = Spectrum([300,500],reflectance = [0.5,0.5], transmittance = [0.5,0.5], 
                     sigma_r = [0.1,0.1], sigma_t = [0.1,0.1])
    assert_equal(spect.wavelength, np.array([300,500]))
    assert_equal(spect.reflectance, np.array([0.5,0.5]))
    assert_equal(spect.transmittance, np.array([0.5,0.5]))
    assert_equal(spect.sigma_r, np.array([0.1,0.1]))
    assert_equal(spect.sigma_t, np.array([0.1,0.1]))
    
    # Test if Spectrum object is created correctly when there is only reflectance 
    spect_refl = Spectrum(500,reflectance = 0.5, sigma_r = 0.1)
    assert_equal(spect_refl.sigma_r, np.array([0.1]))

    # Test if Spectrum object is created correctly when there is only transmittance 
    spect_trans = Spectrum(500,transmittance = 0.5, sigma_t = 0.1)
    assert_equal(spect_trans.sigma_t, np.array([0.1]))
    
def test_Sample():
    # Test if Sample object is created correctly 
    samp = Sample([300,500], 1.5, 1.3)
    assert_equal(samp.particle_index, np.array([1.5, 1.5]))

def test_extend_array():
    assert_equal(extend_array([3,2],2), np.array([3,2]))
    assert_equal(extend_array(3,2), np.array([3,3]))

def test_rescale():
    inarray = np.array([2,3,12])
    outarray = np.array([0,.1,1])
    assert_equal(rescale(inarray),outarray)

def test_find_close_indices():
    big = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
    targets = [np.pi, np.sqrt(50), 10.5]
    expected_output = np.array([2,6,9])
    assert_equal(find_close_indices(big, targets), expected_output)

def test_check_wavelength():
    spect = Spectrum([300,500], reflectance = [0.5,0.5], sigma_r = [0.1,0.1])
    samp = Sample([300,500], 1.5, 1.3)
    assert_equal(check_wavelength(spect, samp), samp.wavelength)
    
def test_convert_dtype():
    assert_equal(np.array([ 1.]), convert_dtype(1))
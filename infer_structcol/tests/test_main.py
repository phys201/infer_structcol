from infer_structcol.main import Spectrum, Sample, extend_array, rescale, find_close_indices, check_wavelength
import numpy as np
from numpy.testing import assert_equal, assert_raises

def test_Spectrum():
    spect = Spectrum([300,500],reflectance = [0.5,0.5], sigma_r = [0.1,0.1])
    assert_equal(spect.wavelength, np.array([300,500]))
    spect1 = Spectrum(500,reflectance = 0.5, sigma_r = 0.1)
    assert_equal(spect1.sigma_r, np.array([0.1]))

def test_Sample():
    samp = Sample([300,500],0.5, 100, 1.5, 1.3)
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
    samp = Sample([300,500],0.5, 100, 1.5, 1.3)
    assert_equal(check_wavelength(spect, samp), samp.wavelength)
    

'''
This file tests functions from io.py.
'''

import os
import shutil
import numpy as np
import infer_structcol
from infer_structcol.io import load_spectrum
from infer_structcol.format_converter import convert_data
from numpy.testing import assert_almost_equal

def test_io():
    # convert the spectrum
    filepath = os.path.dirname(os.path.abspath(__file__))
    direc = os.path.join(filepath, 'test_data', 'simulated_data')
    convert_data(np.array([450,600,800]), 'ref.txt', 'dark.txt', 
                 directory = os.path.join(direc, 'reflection'))
    convert_data(np.array([450,600,800]), 'ref.txt', 'dark.txt', 
                 directory = os.path.join(direc, 'transmission'))
    
    # load the spectrum, creating a spectrum object
    refl_convert_direc = os.path.join(direc,'reflection','converted','0_data_file.txt')
    refl_spectrum = load_spectrum(refl_filepath = refl_convert_direc)
    trans_convert_direc = os.path.join(direc,'transmission','converted','0_data_file.txt')
    trans_spectrum = load_spectrum(trans_filepath = trans_convert_direc)
    
    # check if equal to previously converted data
    assert_almost_equal(refl_spectrum.wavelength[0],450)
    assert_almost_equal(refl_spectrum.reflectance[0],0.68224969055851326)
    assert_almost_equal(refl_spectrum.sigma_r[0],0.022119987764703603)
    assert_almost_equal(trans_spectrum.wavelength[0],450)
    assert_almost_equal(trans_spectrum.transmittance[0],0.2177503094414867668)
    assert_almost_equal(trans_spectrum.sigma_t[0],0.022119987764703603)
    
    refl_spectrum.save(refl_convert_direc)
    trans_spectrum.save(trans_convert_direc)

    try:
        shutil.rmtree(os.path.join(direc, 'reflection', 'converted'))
        shutil.rmtree(os.path.join(direc, 'transmission', 'converted'))
    except:
        pass

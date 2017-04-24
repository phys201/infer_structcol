import os
import shutil
import numpy as np
import infer_structcol
from infer_structcol.io import convert_data,load_spectrum
from numpy.testing import assert_almost_equal

def test_io():
    # convert the spectrum
    filepath = os.path.dirname(os.path.abspath(__file__))
    direc = os.path.join(filepath, 'test_data')
    convert_data(np.array([450,600,800]), 'ref.txt', 'dark.txt', directory = direc) 
    
    # load the spectrum, creating a spectrum object
    convert_direc = os.path.join(direc,'converted','0_data_file.txt')
    spectrum = load_spectrum(convert_direc)
    
    # check if equal to previously converted data
    assert_almost_equal(spectrum.wavelength[0],450)
    assert_almost_equal(spectrum.reflectance[0],0.41846705358309988)
    assert_almost_equal(spectrum.sigma_r[0],0.021908005804960589)
    
    spectrum.save(convert_direc)

    try:
        shutil.rmtree(os.path.join(direc, 'converted'))
    except:
        pass

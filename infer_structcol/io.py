'''
This file contains a function for loading data that has already been converted 
into a usable format.
'''

import os
import numpy as np

from .main import Spectrum

def load_spectrum(**kwargs):
    '''
    Loads a spectrum from a converted data file or saved spectrum
    '''
    # load data from given filepaths
    if 'refl_filepath' in kwargs and 'trans_filepath' in kwargs:
        trans_filedata = np.loadtxt(kwargs['trans_filepath'])
        refl_filedata = np.loadtxt(kwargs['refl_filepath'])
        if np.all(refl_filedata[:,0] != trans_filedata[:,0]):
            raise ValueError("""Wavelengths of transmittance and reflectance 
            spectra do not match""")
        return Spectrum(refl_filedata[:,0], 
                        reflectance = refl_filedata[:,1], 
                        sigma_r = refl_filedata[:,2], 
                        transmittance = trans_filedata[:,1], 
                        sigma_t = trans_filedata[:,2])
    elif 'refl_filepath' in kwargs:
        refl_filedata = np.loadtxt(kwargs['refl_filepath'])
        return Spectrum(refl_filedata[:,0], 
                        reflectance = refl_filedata[:,1], 
                        sigma_r = refl_filedata[:,2])
    elif 'trans_filepath' in kwargs:
        trans_filedata = np.loadtxt(kwargs['trans_filepath'])
        return Spectrum(trans_filedata[:,0],
                        transmittance = trans_filedata[:,1],
                        sigma_t = trans_filedata[:,2])
    else:
        raise ValueError("""You must enter one or both keyword arguments 
        refl_filepath or trans_filepath""")
    
    
    
    

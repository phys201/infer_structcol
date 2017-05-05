'''
This file contains functions for loading data and converting it into a usable 
format
'''

import os
import numpy as np
import glob
import pandas as pd

from .main import Spectrum, find_close_indices

def load_exp_data(wavelen, ref_file, dark_file, directory = ''):
    '''
    Loads spectrum data for a given set of wavelengths 
    All data must consist of two, tab-separated columns. The first is the 
    wavelength, and the second is the normalized reflection or transmission 
    fraction.
    
    Parameters
    ----------
    wavelen: array-like
        wavelengths for which spectrum values should be loaded
    ref_file: string
        file name of the reference measurement.
    dark_file: string
        file name of the dark measurement. .
    
    Returns
    -------
    dark: pandas array
        array with columns for wavelength and intensity from dark measurement
    ref: pandas array
        array with columns for wavelength and intensity from ref measurement
    spec: numpy array
        multidimentional array where first dimension is number of spectrum 
        measurements and second dimension is wavlengths
    
    '''
    filelist = glob.glob(os.path.join(directory,'*.txt'))
    spec = np.array([])
    dark = None
    ref = None

    # iterate through files in directory, finding reference, dark, and spectra
    for filename in filelist:
        if filename == os.path.join(directory,dark_file):
            dark = pd.read_table(filename, 
                                 names = ['wavelength','intensity']).dropna().reset_index(drop = True)
            dark = dark.apply(pd.to_numeric)
        elif filename == os.path.join(directory,ref_file):
            ref = pd.read_table(filename, 
                                names = ['wavelength','intensity']).dropna().reset_index(drop = True)
            ref = ref.apply(pd.to_numeric)
        else:
            spec = np.append([spec], [pd.read_table(filename, 
                             names = ['wavelength','intensity']).dropna().reset_index(drop = True).intensity])
    
    if dark is None or ref is None:
        raise IOError("Could not find normalization files. Check your path names.")

    # find the indices of the wavelengths of interest in the data
    wl_ind = find_close_indices(dark['wavelength'].values, wavelen)

    # take only the wavelengths of interest
    spec = spec.reshape((len(filelist)-2,len(dark),))
    spec = spec[:,wl_ind]
    dark = dark.iloc[wl_ind]
    ref = ref.iloc[wl_ind]
        
    return ref, dark, spec

def calc_norm_spec(ref, dark, spec):
    '''
    Calculates a normalized spectrum.
    
    Parameters
    -------
    dark: pandas array
        array with columns for wavelength and intensity from dark measurement
    ref: pandas array
        array with columns for wavelength and intensity from ref measurement
    spec: numpy array
        multidimentional array where first dimension is number of spectrum 
        measurements and second dimension is wavlengths
        
    Returns
    -------
    multidimentional array of normalized spectra, where first dimension 
    corresponds to number of spectra and second dimension corresponds to number
    of wavelengths 
    
    '''
    return (spec-dark['intensity'])/(ref['intensity']-dark['intensity'])
    

def convert_data(wavelen, ref_file, dark_file, directory = ''):
    '''
    Write loaded experimental data to file in columns wavelength, 
    normalized intensity, and standard deviation, respectively.
    
    Parameters
    -------
    wavelen: array-like
        wavelengths for which spectrum values should be loaded
    dark: pandas array
        array with columns for wavelength and intensity from dark measurement
    ref: pandas array
        array with columns for wavelength and intensity from ref measurement
    spec: numpy array
        multidimentional array where first dimension is number of spectrum 
        measurements and second dimension is wavlengths
    single_spec: array-like
        array of intensity data for singe data set of interest
    '''

    ref, dark, spec = load_exp_data(wavelen, ref_file, dark_file, directory)
    norm_spec = np.zeros(spec.shape)
    for i in range(spec.shape[0]):
        norm_spec[i,:] = calc_norm_spec(ref, dark, spec[i,:])
    stdev = np.std(norm_spec, axis = 0, ddof = 1)

    directory = os.path.join(directory, 'converted')
    if not os.path.isdir(directory):
        os.mkdir(directory)
    for i in range(spec.shape[0]):
        np.savetxt(os.path.join(directory,str(i)+'_data_file.txt'), 
                   np.c_[wavelen, norm_spec[i,:], stdev])
    
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
    
    
    
    

import os
import numpy as np
import glob
import pandas as pd

from .main import Spectrum

def load_exp_data(wavelen, ref_file, dark_file, directory = ''):
    '''
    loads spectrum data for a given set of wavelengths 
    All data must consist of two, tab-separated columns. The fist is the 
    wavelength, and the second is the intensity.
    
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
            dark = pd.read_table(filename, names = ['wavelength', 'intensity']).dropna().reset_index(drop = True)
            dark = dark.apply(pd.to_numeric)
        elif filename == os.path.join(directory,ref_file):
            ref = pd.read_table(filename, names = ['wavelength', 'intensity']).dropna().reset_index(drop = True)
            ref = ref.apply(pd.to_numeric)
        else:
            spec = np.append([spec], [pd.read_table(filename, names = ['wavelength', 'intensity']).dropna().reset_index(drop = True).intensity])
    
    if dark is None or ref is None:
        raise IOError("Could not find normalization files. Check your path names."

    # find the indices of the wavelengths of interest in the data
    wl_ind = []
    for i, wl in enumerate(wavelen):
        wli = np.where(abs(dark['wavelength']-wl) == np.min(abs(dark['wavelength']-wl)))[0]
        if len(wli>1):
            wli = wli[0]
        wl_ind.append(int(wli))

    # take only the wavelengths of interest
    spec = spec.reshape((len(filelist)-2,len(dark),))
    spec = spec[:,wl_ind]
    dark = dark.iloc[wl_ind]
    ref = ref.iloc[wl_ind]
        
    return ref, dark, spec

def calc_norm_spec(ref, dark, spec):
    '''
    calculates a normalized spectrum 
    
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
    write loaded experimental data to file in columns wavelength, 
    normalized intensity, and standard deviation, respectively
    
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

    ref, dark, spec = load_exp_data(wavelen, ref_file, dark_file, directory = '')
    norm_spec = np.zeros(spec.shape)
    for i in range(spec.shape[0]):
        norm_spec[i,:] = calc_norm_spec(ref, dark, spec[i,:])
    stdev = np.std(norm_spec, axis = 0, ddof = 1)

    directory = os.path.join(directory, 'converted')
    if not os.path.isdir(directory):
        os.mkdir(directory)
    for i in range(spec.shape[0]):
        np.savetxt(os.path.join(directory,str(i)+'_data_file.txt'), np.c_[wavelen, norm_spec[i,:], stdev])
    
def load_spectrum(filepath):
    '''
    Loads a spectrum from a converted data file or saved spectrum
    '''
    filedata = np.loadtxt(filepath)
    return Spectrum(filedata[:,0], filedata[:,1], filedata[:,2])

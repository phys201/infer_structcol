'''
This file contains functions for loading experimental reflectance and transmittance
data collected with some commonly used instruments in the field, and converts them 
into a usable format. 

The instruments for which format conversion is available are:

Ocean Optics HR2000+ Spectrometer, Ocean Optics SpectraSuite software 
Agilent Technologies Cary 7000 UMS 

'''
import os
import numpy as np
import glob
import pandas as pd

from .main import find_close_indices, find_filenames


###############################################################################
# For data taken with Ocean Optics HR2000+ Spectrometer, Ocean Optics SpectraSuite software 

def convert_data(wavelen, ref_file, dark_file, directory = ''):
    '''
    Write loaded experimental data to file in columns wavelength, normalized 
    intensity, and standard deviation, respectively. This function is tailored
    to data collected with Ocean Optics HR2000+ Spectrometer, Ocean Optics 
    SpectraSuite software.
    
    Parameters
    ----------
    wavelen: array-like
        wavelengths for which spectrum values should be loaded
    dark_file: pandas array
        array with columns for wavelength and intensity from dark measurement
    ref_file: pandas array
        array with columns for wavelength and intensity from ref measurement

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
        measurements and second dimension is wavelengths
    
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
    ----------
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



###############################################################################
# For data taken with Agilent Technologies Cary 7000 UMS 

def convert_data_csv(wavelength, directory):
    '''
    Write experimental data in .csv to file in columns of wavelength, normalized 
    intensity, and standard deviation, respectively. This function is tailored 
    to the experimental data that is collected with Agilent Technologies 
    Cary 7000 UMS integrating sphere. It assumes that the measured wavelengths
    are 800 to 400 in decreasing order. 
    
    Parameters
    ----------
    wavelength: ndarray
        wavelengths for which spectrum values should be loaded
    directory : str
        directory where data is stored
    
    '''
    filenames = find_filenames(directory)

    # load data lines of each csv file using pandas
    df = pd.read_csv(directory + '/' + filenames[0])[1:402]

    # find the number of samples in the file by counting how many variables
    # containing the string "sample" there are
    number_samples = df.filter(regex='sample').shape[1]

    # create matrices that contain data, wavelengths, and st dev of all the 
    # samples. Rows are wavelengths, columns are samples
    data = np.zeros([len(df),number_samples])
    for i in np.arange(1, number_samples+1):
        sample_name = 'sample' + str(i)
        data[:,i-1] = df.iloc[:, df.columns.get_loc(sample_name)+1]
        data[:,i-1] = data[:,i-1][::-1] / 100
            
    full_wavelength = np.array([int(wl) for wl in df['sample1'][::-1]])
    wl_ind = find_close_indices(full_wavelength, wavelength)
    data = data[wl_ind,:]
    std_dev = np.std(data, axis = 1, ddof = 1)

    # save the wavelengths, data, and st dev into converted txt files
    directory = os.path.join(directory, 'converted')
    if not os.path.isdir(directory):
        os.mkdir(directory)
    
    for i in range(number_samples):
        np.savetxt(os.path.join(directory, str(i)+'_data_file.txt'), 
                   np.c_[wavelength, data[:,i], std_dev])
        
###############################################################################
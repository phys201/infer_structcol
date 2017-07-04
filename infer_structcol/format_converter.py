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

def convert_data_csv(directory, min_wavelength, max_wavelength, header='sample'):
    '''
    Write experimental data in .csv to .txt file in columns of wavelength, 
    normalized intensity, and standard deviation, respectively. This function 
    is tailored to the experimental data that is collected with Agilent 
    Technologies Cary 7000 UMS integrating sphere. This function assumes the 
    default data output from the software, which consists of each wavelength 
    column headed by the string 'sample' (e.g. 'sample1', 'sample2', etc), 
    followed by a column with the reflectance (in %), transmittance (in %), 
    or absorbance data (fraction). 
    
    Parameters
    ----------
    directory : str
        directory where data is stored
    min_wavelength : int
        minimum wavelength recorded in the .csv data
    max_wavelength : int
        maximum wavelength recorded in the .csv data
    header : str (optional)
        common header of the columns in the .csv file corresponding to each 
        sample measurement (e.g. 'sample')   
    '''
    # find all the files in the directory that are .csv
    filenames = find_filenames(directory)
    
    for fn in np.arange(len(filenames)):
        # load data lines of each csv file using pandas, skip the second row 
        # that contains the wavelength and data headers
        df = pd.read_csv(os.path.join(directory,filenames[fn]), skiprows=[1])

        # find the rows in the .csv that correspond to the data and trim the 
        # dataframe to contain only the data
        min_wavelength_row = np.where(df[df.columns[0]]==str(min_wavelength))[0]
        max_wavelength_row = np.where(df[df.columns[0]]==str(max_wavelength))[0]
        last_data_row = int(max(min_wavelength_row, max_wavelength_row))
        df = df[:last_data_row+1]
        
        # make up a list of the samples by looking for the string 'sample' in 
        # the header of the spreadsheet columns
        sample_list = df.filter(regex=header).columns.values

        # make new dataframes for the spectral data of the samples and the 
        # corresponding wavelengths
        sample_data = np.zeros([len(df),len(sample_list)])
        sample_wavelengths = np.zeros([len(df),len(sample_list)])
        for i in np.arange(len(sample_list)):
            # get the wavelengths and data for each sample
            sample_wavelengths[:,i] = df.iloc[:, df.columns.get_loc(sample_list[i])]
            sample_data[:,i] = df.iloc[:, df.columns.get_loc(sample_list[i])+1]
    
            # if the wavelengths are in decreasing order, then revert the order 
            # of the wavelengths and the data
            if sample_wavelengths[0,i] > sample_wavelengths[-1,i]:
                sample_wavelengths[:,i] = sample_wavelengths[:,i][::-1] 
                sample_data[:,i] = sample_data[:,i][::-1] 
            
            # if the data are in percentage (not normalized from 0 to 1), then
            # divide by 100 
            if any(data > 1 for data in sample_data[:,i]):   
                sample_data[:,i] = sample_data[:,i] / 100
                
        sample_data = pd.DataFrame(data=sample_data, columns=sample_list)
        sample_wavelengths = pd.DataFrame(data=sample_wavelengths, columns=sample_list)

        # calculate standard deviation
        std_dev = sample_data.std(axis = 1, ddof = 1)
    
        # save the wavelengths, data, and st dev into converted txt files
        # if there is more than one .csv file in the directory, then save into 
        # different folders for each .csv file
        if len(filenames) > 1:
            directory_save = os.path.join(directory, 
                                          filenames[fn].replace('.csv', '')+'_converted')
        else: 
            directory_save = os.path.join(directory, 'converted')
        
        if not os.path.isdir(directory_save):
            os.mkdir(directory_save)
    
        for i in range(len(sample_list)):
            np.savetxt(os.path.join(directory_save, str(i)+'_data_file.txt'), 
                       np.c_[sample_wavelengths[sample_list[i]], 
                             sample_data[sample_list[i]], std_dev])       
###############################################################################
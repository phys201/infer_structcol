'''
This file contains classes used to handle data and a few simple functions for 
manipulating numpy arrays. 
'''

import pandas as pd
import numpy as np 
from os import listdir

class Spectrum(pd.DataFrame):
    '''
    Stores Spectrum data in a pandas DataFrame with a few extra methods and 
    functionality.
    
    Parameters
    -------
    wavelength: array of length N
        wavelengths of spectrum
    reflectance: array of length N
        reflectance values at each wavelength
    reflectance_sigma: array of length N
        uncertainty on each reflection value
    transmittance: array of length N
        reflectance values at each wavelength
    tranmittance_sigma: array of length N
        uncertainty on each reflection value
    '''

    def __init__(self, wavelength, **kwargs):
        if 'reflectance' in kwargs and 'transmittance' in kwargs:
            super(Spectrum,self).__init__({"wavelength":convert_dtype(wavelength), 
            "reflectance":convert_dtype(kwargs['reflectance']), 
            "sigma_r":convert_dtype(kwargs['sigma_r']),
            "transmittance":convert_dtype(kwargs['transmittance']), 
            "sigma_t":convert_dtype(kwargs['sigma_t'])})
        elif 'reflectance' in kwargs:
            super(Spectrum,self).__init__({"wavelength":convert_dtype(wavelength), 
            "reflectance":convert_dtype(kwargs['reflectance']), 
            "sigma_r":convert_dtype(kwargs['sigma_r'])})       
        else:
            super(Spectrum,self).__init__({"wavelength":convert_dtype(wavelength), 
            "transmittance":convert_dtype(kwargs['transmittance']), 
            "sigma_t":convert_dtype(kwargs['sigma_t'])}) 
            
    

    @property
    def wavelength(self):
        return self['wavelength'].values
    @property
    def reflectance(self):
        return self['reflectance'].values
    @property
    def sigma_r(self):
        return self['sigma_r'].values
    @property
    def transmittance(self):
        return self['transmittance'].values
    @property
    def sigma_t(self):
        return self['sigma_t'].values
    
    def save(self, filepath):
        if not filepath[-4:] =='.txt':
            filepath = filepath + '.txt.'
        if 'reflectance' in self.keys() and 'transmittance' in self.keys():
            np.savetxt(filepath, np.c_[self.wavelength, self.reflectance, self.sigma_r, 
                                       self.transmittance, self.sigma_t])
        elif 'reflectance' in self.keys():
            np.savetxt(filepath, np.c_[self.wavelength, self.reflectance, self.sigma_r])
        else:
            np.savetxt(filepath, np.c_[self.wavelength, self.transmittance, self.sigma_t])

class Sample:
    '''
    Stores information about a physical sample.
    
    Parameters
    -------
    wavelength: array of length N
        wavelengths corresponding to refractive indices
    particle_index: array of length N or scalar
        refractive index of particles at each wavelength
    matrix_index: array of length N or scalar
        refractive index of matrix at each wavelength
    medium_index: array of length N or scalar
        refractive index of propagation medium at each wavelength
    incident_angle: scalar
        angle of incident light on the sample
    '''
    def __init__(self, wavelength, particle_index, 
                 matrix_index, medium_index=1, incident_angle=0):
        self.wavelength = convert_dtype(wavelength)   # in nm

        self.particle_index = extend_array(particle_index, len(self.wavelength))
        self.matrix_index = extend_array(matrix_index, len(self.wavelength))
        self.medium_index = extend_array(medium_index, len(self.wavelength))
        self.incident_angle = incident_angle

def extend_array(val, n):
    '''
    Returns an array of val of length n.
    
    Parameters
    -------
    val: array of length n or scalar
        value to be filled into array
    n: int
        desired length of array
    '''
    val = convert_dtype(val)
    if len(val) == n:
        return val
    elif len(val) == 1:
        return np.repeat(val, n)
    else:
        raise IndexError("Arrays are of different lengths. Cannot interpret.")

def rescale(inarray):
    '''
    Rescales an array so all values are between 0 and 1.
    '''
    maxval = np.max(inarray)
    minval = np.min(inarray)
    if maxval == minval:
        return 0
    return (inarray-minval)/(maxval-minval)

def find_close_indices(biglist, targets):
    '''
    Find the positions (indices) of targets within a larger array.
    
    Parameters
    -------
    biglist: list-like
        contains values to be searched over
    targts: list-like
        values of interest to be searched for
    '''
    target_ind = []
    for i, target in enumerate(np.array(targets)):
        diffs = abs(np.array(biglist)-target)
        curr_ind = np.where(diffs == np.min(diffs))[0][0]
        target_ind.append(curr_ind)
    return target_ind

def check_wavelength(obj1, obj2):
    '''
    Make sure Spectrum or Sample objects have the same 'wavelength' attributes.
    '''
    if np.all(obj1.wavelength == obj2.wavelength):
        return obj1.wavelength
    raise ValueError("Wavelength mismatch.")

def convert_dtype(inval):
    '''
    Converts value to double precision float.
    '''
    if np.isscalar(inval):
        inval = [inval]
    return np.array(inval).astype('float64')

def find_filenames(directory, suffix=".csv" ):
    '''
    This function lists all the csv (by default) files in a directory.
    
    Parameter
    ---------
    directory : str
        directory where csv files are stored
    
    '''
    filenames = listdir(directory)
    return [filename for filename in filenames if filename.endswith(suffix)]
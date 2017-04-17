import pandas as pd
import numpy as np 

class Spectrum(pd.DataFrame):
    def __init__(self, reflectance_file, transmission_file=None)
        #pd.read_table or similar

        if transmission_file is not None:
            raise NotImplementedError()

    @property
    def wavelength(self):
        return self['wavelength'].values
    @property
    def meas(self):
        return self['measurement'].values
    @property
    def sigma(self):
        return self['sigma'].values

class Sample:
    def __init__(self, particle_size, thickness, particle_index, matrix_index, medium_index=1, incident_angle=0):
        self.particle_size = particle_size # can we do something clever here with units? maybe using pint?
        self.thickness = thickness # again with the units
        self.matrix_index = matrix_index
        self.medium_index = medium_index
        self.incident_angle = incident_angle 

def rescale(inarray):
    # Rescale values in an array to be between 0 and 1
    maxval = np.max(inarray)
    minval = np.min(inarray)
    return (inarray-minval)/(maxval-minval)

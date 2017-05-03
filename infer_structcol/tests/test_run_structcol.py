'''
This file tests functions from run_structcol.py.
'''

import infer_structcol
import numpy as np
from numpy.testing import assert_equal, assert_almost_equal

def test_structcol_import():
    # Test that montecarlo.py is imported and Trajectory object is imported correctly
    from structcol import montecarlo as mc
    dummy_trajectory = mc.Trajectory("position", "direction", "weight")
    assert_equal(dummy_trajectory.position, "position")
    
def test_run_structcol():
    # Test that structcol package is imported and reflectance calculation is correct
    import structcol as sc
    from infer_structcol.run_structcol import calc_refl_trans
    from infer_structcol.main import Sample
    
    wavelength = np.array([400., 500.]) 
    particle_radius = 150.
    thickness = 100.
    particle_index = np.array([1.40, 1.41]) 
    matrix_index = np.array([1.0, 1.0]) 
    volume_fraction = 0.5
    incident_angle = 0.0
    medium_index = np.array([1.0, 1.0]) 
    
    spectrum = calc_refl_trans(volume_fraction, Sample(wavelength, particle_radius, 
                                                          thickness, particle_index, 
                                                          matrix_index, medium_index, 
                                                          incident_angle), seed=1)

    outarray = np.array([0.85201,  0.7326])
    assert_almost_equal(spectrum.reflectance, outarray, decimal=5)

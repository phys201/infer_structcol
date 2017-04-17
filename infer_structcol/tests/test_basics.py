import infer_structcol
import os
import numpy as np
from numpy.testing import assert_equal

def test_structcol_import():
    from structcol import montecarlo as mc
    dummy_trajectory = mc.Trajectory("position", "direction", "weight", "nevents")
    assert_equal(dummy_trajectory.position, "position")

def test_rescale():
    from infer_structcol.main import rescale
    inarray = np.array([2,3,12])
    outarray = np.array([0,.1,1])
    assert_equal(rescale(inarray),outarray)

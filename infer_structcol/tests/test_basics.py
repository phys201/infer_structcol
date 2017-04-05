import infer_structcol
import os
from numpy.testing import assert_equal

def test_structcol_import():
    from structcol import montecarlo as mc
    dummy_trajectory = mc.Trajectory("position", "direction", "weight", "nevents")
    assert_equal(dummy_trajectory.position, "position")




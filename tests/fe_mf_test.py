
import pytest
import numpy
import math
import pyhsi as hsi


def test_fe_mf_single_ped():

    # Run program
    numSteps = 5000

    m = 73.85
    k = 14.11e3
    xi = 0.3
    crowd = hsi.TestCrowd(m, xi * 2 * math.sqrt(k * m), k, 2, 0, 0, 1.25, 0)

    # run fe_mf_crowd_solve
    fe_mf_solver = hsi.FeMfSolver(crowd, numSteps)
    maxRMS = fe_mf_solver.maxRMS
    assert maxRMS == pytest.approx(1.222956, 0.0001)


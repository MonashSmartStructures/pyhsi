# from .crowd import Crowd
from crowd import Crowd
from beam import Beam


def fe_mf():

    # Guess number of steps
    nSteps = 5000

    # Create crowd and beam objects
    crowd = Crowd(0.5, 100, 2, 0.1)
    beam = Beam()

    # Create lamda
    lamda = [0]*(2*beam.numElements)

    # run fe_mf_crowd_solve
    t, q, dq, ddq = fe_mf_solve(nSteps, crowd, beam)


def fe_mf_solve(nSteps, crowd, beam):
    # Filler return values
    t = 0
    q = 0
    dq = 0
    ddq = 0

    nBDOF = 2*(beam.numElements+1)

    # Assemble MCK

    # Set up time vector

    # Assemble force and global matrices
    M, C, K, F = fe_mf_crowd(t, crowd, beam)

    # Run Newmark

    return t, q, dq, ddq


def fe_mf_crowd(t, crowd, beam):
    # Filler return values
    M = 0
    C = 0
    K = 0
    F = 0

    # Apply constraints to bridge

    # Initialise global matrices

    # Shape function zero matrices

    # For each pedestrian
    # loop

    return M, C, K, F


fe_mf()

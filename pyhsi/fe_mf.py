import math

from crowd import Crowd
from beam import Beam


def fe_mf():

    # Guess number of steps
    nSteps = 5000

    # Create crowd and beam objects
    crowd = Crowd(0.5, 100, 2, 0.1)
    # beam = Beam()

    # Create lamda
    # lamda = [0]*(2*beam.numElements)

    # run fe_mf_crowd_solve
    fe_mf_solver = FeMfSolver(crowd, nSteps)

    # t, q, dq, ddq = fe_mf_solve(nSteps, crowd, beam)


class FeMfSolver:

    def __init__(self, crowd, nSteps):
        self.crowd = crowd
        self.beam = Beam()
        self.nSteps = nSteps

        self.Mb = 0
        self.Cb = 0
        self.Kb = 0

        self.t = 0
        self.dT = 0

        self.M = 0
        self.C = 0
        self.K = 0
        self.F = 0

        self.q = 0
        self.dq = 0
        self.ddq = 0

        self.assembleMCK()

    def assembleMCK(self):

        nElements = self.beam.numElements

        nDOF = 2*(nElements + 1)
        elementalMassMatrix, elementalStiffnessMatrix = self.beam.beamElement()

        M = [[0]*nDOF]*nDOF
        C = [[0]*nDOF]*nDOF
        K = [[0]*nDOF]*nDOF

        # Assemble elements, noting beam is prismatic
        for i in range(nElements):
            ni = 2*(i-1)+1
            nj = 2*(i-1)+4
            M[ni:nj][ni:nj] += elementalMassMatrix
            K[ni:nj][ni:nj] += elementalStiffnessMatrix

        # Apply constraints before estimating modal properties
        # Save to temp variables to keep M, K
        # Mt, Ct, Kt = constraints(M, C, K)

        self.Mb = 0
        self.Cb = 0
        self.Kb = 0

    def createTimeVector(self):
        self.t, self.dT = simTime(self.crowd, self.beam, self.nSteps)

    def assembleMatrices(self):

        self.M = 0
        self.C = 0
        self.K = 0
        self.F = 0

    def solver(self):
        # Runs Newmark-Beta integration for MDOF systems
        self.q = 0
        self.dq = 0
        self.ddq = 0


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


def simTime(crowd, beam, nSteps):
    t = 0
    dT = 0
    return t, dT


def constraints(M, C, K):
    # def imposeRestraint(A, dof):


fe_mf()

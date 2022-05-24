import math

from crowd import Crowd, testCrowd
from beam import Beam
import math
import numpy as np


def fe_mf():

    # Guess number of steps
    nSteps = 5000

    # Create crowd and beam objects
    crowd = testCrowd(80,650,21500,2.10,math.pi,0,1.51,0)
    #crowd = Crowd(0.5, 100, 2, 0.1)
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

        #self.assembleMCK()
        self.createTimeVector()

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

def simTime(crowd, beam, nSteps): #Returns the simulation time frame for a given pace and stride vector
    f = 1 / (2 * math.pi) * (math.pi/beam.length)**2 * math.sqrt(beam.EI/beam.linearMass)
    Period = 1 / f
    dTmax = 0.02 * Period
    pVel = crowd.pVel
    Toff = (-crowd.pLoc + beam.length) / pVel
    Tend = 1.1 * Toff
    dT = Tend / (nSteps)
    dT = min(dT,dTmax)
    t = np.arange(0, Tend, dT)      #Rounding error created by differing precision in Python vs MATLAB
    return t, dT                    #may have caused discrepancies in output values.

def constraints(M, C, K):
    return
 #def imposeRestraint(A, dof):


fe_mf()


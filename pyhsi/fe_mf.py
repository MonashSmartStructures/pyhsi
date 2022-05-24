import math
import numpy as np

from scipy.linalg import eig
from scipy.linalg import eigh


from crowd import Crowd, testCrowd
from beam import Beam


def fe_mf():

    # Guess number of steps
    nSteps = 5000

    # Create crowd and beam objects
    crowd = testCrowd(80, 650, 21500, 2.10, math.pi, 0, 1.51, 0)
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

        self.assembleMCK()
        self.createTimeVector()

    def assembleMCK(self):

        nElements = self.beam.numElements

        nDOF = 2*(nElements + 1)
        RDOF = [0, nDOF-2]  # Should this be nDOF-1 so that the last column is used not 2nd last

        elementalMassMatrix, elementalStiffnessMatrix = self.beam.beamElement()

        M = np.array([[0]*nDOF]*nDOF, dtype='f')
        C = M.copy()
        K = M.copy()

        # Assemble elements, noting beam is prismatic
        for i in range(1, nElements+1):
            ni = 2*(i-1)
            nj = 2*(i-1)+4
            M[ni:nj, ni:nj] += elementalMassMatrix
            K[ni:nj, ni:nj] += elementalStiffnessMatrix

        # Apply constraints before estimating modal properties
        # Save to temp variables to keep M, K
        Mt = M.copy()
        Ct = C.copy()
        Kt = K.copy()
        Mt, Ct, Kt = constraints(RDOF, Mt, Ct, Kt)

        phi, w = modal(Mt, Kt)

        alpha, beta = rayleighCoeffs(w, self.beam.modalDampingRatio, self.beam.nHigh)

        C = alpha*M + beta*K

        self.Mb = M
        self.Cb = C
        self.Kb = K

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
    # Returns the simulation time frame for a given pace and stride vector
    f = 1 / (2 * math.pi) * (math.pi/beam.length)**2 * math.sqrt(beam.EI/beam.linearMass)
    Period = 1 / f
    dTmax = 0.02 * Period
    pVel = crowd.pVel
    Toff = (-crowd.pLoc + beam.length) / pVel
    Tend = 1.1 * Toff
    dT = Tend / (nSteps)
    dT = min(dT, dTmax)
    t = np.arange(0, Tend, dT)      #Rounding error created by differing precision in Python vs MATLAB
    return t, dT                    #may have caused discrepancies in output values.


def constraints(RDOF, M, C, K):
    def imposeRestraint(A, dof):
        A[dof] = 0          # column
        A[:, dof] = 0       # row
        A[dof, dof] = 1     # diagonal

        return A

    for i in RDOF:
        dof = i
        M = imposeRestraint(M, dof)
        C = imposeRestraint(C, dof)
        K = imposeRestraint(K, dof)

    return M, C, K


def modal(M, K):

    lam, phi = eigh(K, M)
    n, m = K.shape
    omega = np.sqrt(lam)

    return phi, omega


def rayleighCoeffs(w, modalDampingRatio, nHigh):

    wr = [i for i in w if i > 1.01]
    wi = wr[0]
    wj = wr[nHigh-1]

    alpha = modalDampingRatio*(2*wi*wj)/(wi+wj)
    beta = modalDampingRatio*2/(wi+wj)

    return alpha, beta


fe_mf()

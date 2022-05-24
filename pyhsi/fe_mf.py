import math
import numpy as np
import pandas as pd

from scipy.linalg import eig
from scipy.linalg import eigh

from crowd import Crowd, TestCrowd
from beam import Beam


def fe_mf():
    # Guess number of steps
    nSteps = 5000

    # Create crowd and beam objects
    crowd = TestCrowd(80, 650, 21500, 2.10, math.pi, 0, 1.51, 0)
    # crowd = Crowd(0.5, 100, 2, 0.1)
    # beam = Beam()

    # Create lamda
    # lamda = [0]*(2*beam.numElements)

    # run fe_mf_crowd_solve
    fe_mf_solver = FeMfSolver(crowd, nSteps)

    # t, q, dq, ddq = fe_mf_solve(nSteps, crowd, beam)


class FeMfSolver:
    # This may eventually be a general solver for all model types

    def __init__(self, crowd, nSteps):
        self.crowd = crowd
        self.beam = Beam()
        self.nSteps = nSteps

        self.Mb = 0
        self.Cb = 0
        self.Kb = 0

        self.t = []
        self.dT = 0

        self.M = []
        self.C = []
        self.K = []
        self.F = []

        self.q = 0
        self.dq = 0
        self.ddq = 0

        self.assembleMCK()
        self.createTimeVector()
        self.assembleMatrices()
        self.solver()
        print(self.q)
        print(self.dq)
        print(self.ddq)

    def assembleMCK(self):

        nElements = self.beam.numElements

        nDOF = self.beam.nDOF
        RDOF = self.beam.RDOF

        elementalMassMatrix, elementalStiffnessMatrix = self.beam.beamElement()

        M = np.array([[0] * nDOF] * nDOF, dtype='f')
        C = M.copy()
        K = M.copy()

        # Assemble elements, noting beam is prismatic
        for i in range(1, nElements + 1):
            ni = 2 * (i - 1)
            nj = 2 * (i - 1) + 4
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

        C = alpha * M + beta * K

        self.Mb = M
        self.Cb = C
        self.Kb = K

    def createTimeVector(self):
        self.t, self.dT = simTime(self.crowd, self.beam, self.nSteps)

    def assembleMatrices(self):
        # fe_mf_crowd in matlab

        g = 9.81
        nDOF = self.beam.nDOF
        nBDOF = self.beam.nBDOF
        RDOF = self.beam.RDOF

        # Apply constraints to bridge
        self.Mb, self.Cb, self.Kb = constraints(RDOF, self.Mb, self.Cb, self.Kb)

        # Initialise global matrices
        M = self.Mb
        C = self.Cb
        K = self.Kb
        nStep = len(self.t)
        F = np.array([[0] * nDOF] * nStep, dtype='f')

        # Shape function zero matrices
        Ng0 = np.array([0] * nBDOF, dtype='f')  # Zero vector
        elementLength = self.beam.length / self.beam.numElements

        # For each pedestrian
        nPed = 1

        # Pedestrian force, position and matrices
        x, Ft = pedForce(self.t, self.crowd)

        # For each position, assemble force into matrix
        for j in range(nStep):
            N = globalShapeFunction(x[j], self.beam.length, self.beam.numElements, elementLength, nBDOF, RDOF, Ng0)
            F[j] += N * Ft[j]

        self.M = M
        self.C = C
        self.K = K
        # self.F = F

        # Import F from xlsx file
        self.F = np.genfromtxt('F.csv', delimiter=',')

    def solver(self):
        # Runs Newmark-Beta integration for MDOF systems
        u0 = np.zeros(self.beam.nDOF)
        du0 = np.zeros(self.beam.nDOF)
        self.q, self.dq, self.ddq = newMark(self.t, self.M, self.C, self.K, self.F, u0, du0)


def newMark(t, M, C, K, F, u0, du0):
    # Select algorithm parameters
    gamma = 0.5
    beta = 0.25
    nDOF = len(M)
    n = len(t)

    if n == 1:
        h = t
    else:
        h = t[1] - t[0]

    # Effective stiffness and other matrices
    Keff = K + (gamma / (beta * h)) * C + (1 / (beta * h ** 2)) * M
    iKeff = np.linalg.inv(Keff)
    a = (1 / (beta * h)) * M + (gamma / beta) * C
    b = (1 / (2 * beta)) * M + h * (gamma / (2 * beta) - 1) * C

    # Force increments
    dF = np.zeros((n, nDOF))
    asdf = np.diff(F, axis=0)
    dF[0:n - 1] = np.diff(F, axis=0)
    dF[-1] = 0

    # Initial acceleration
    ddu0 = np.linalg.inv(M) * np.transpose(F[0, :]) - C * du0 - K * u0

    # Initial Conditions and output matrices
    u = np.zeros((n, nDOF))
    du = np.zeros((n, nDOF))
    ddu = np.zeros((n, nDOF))
    u[0] = u0
    du[0] = du0
    ddu[0] = ddu0

    # loop for all time steps
    for i in range(1, n - 1):
        dFeff = np.transpose(dF[i - 1]) + a * np.transpose(du[i - 1]) + b * np.transpose(ddu[i - 1])
        delta_u = iKeff * dFeff
        delta_du = (gamma / (beta * h)) * delta_u - (gamma / beta) * np.transpose(du[i - 1]) + \
                   h * (1 - gamma / (2 * beta)) * np.transpose(ddu[i - 1])
        delta_ddu = (1 / (beta * h ** 2)) * delta_u - (1 / (beta * h)) * np.transpose(du[i - 1]) - \
                    (1 / (2 * beta)) * np.transpose(ddu[i - 1])

        u[i] = u[i - 1] + np.transpose(delta_u)
        du[i] = du[i - 1] + np.transpose(delta_du)
        ddu[i] = ddu[i - 1] + np.transpose(delta_ddu)

    return u, du, ddu


def simTime(crowd, beam, nSteps):
    # Returns the simulation time frame for a given pace and stride vector
    f = 1 / (2 * math.pi) * (math.pi / beam.length) ** 2 * math.sqrt(beam.EI / beam.linearMass)
    Period = 1 / f
    dTmax = 0.02 * Period
    pVel = crowd.pVel
    Toff = (-crowd.pLoc + beam.length) / pVel
    Tend = 1.1 * Toff
    dT = Tend / (nSteps)
    dT = min(dT, dTmax)
    t = np.arange(0, Tend, dT)  # Rounding error created by differing precision in Python vs MATLAB
    return t, dT  # may have caused discrepancies in output values.


def constraints(RDOF, M, C, K):
    def imposeRestraint(A, dof):
        A[dof] = 0  # column
        A[:, dof] = 0  # row
        A[dof, dof] = 1  # diagonal

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
    wj = wr[nHigh - 1]

    alpha = modalDampingRatio * (2 * wi * wj) / (wi + wj)
    beta = modalDampingRatio * 2 / (wi + wj)

    return alpha, beta


def pedForce(t, ped):
    # Question: What are all the commented out parts in matlab ped_force
    g = 9.81

    pMass = ped.pMass
    pPhase = ped.pPhase
    pPace = ped.pPace
    pLoc = ped.pLoc
    pVel = ped.pVel

    W = pMass * g
    x = pLoc + pVel * t  # Position of Pedestrian at each time t

    # Young
    eta = np.array([0.41 * (pPace - 0.95), 0.069 + 0.0056 * pPace, 0.033 + 0.0064 * pPace, 0.013 + 0.0065 * pPace])
    phi = [0] * 4

    # Now assemble final force, and include weight
    N = len(eta)  # No. of additional terms in harmonic series
    F0 = W * np.insert(eta, 0, 1)  # Force amplitudes (constant amplitude for 1)
    beta = 2 * math.pi * pPace * np.array([i for i in range(N + 1)])  # Frequencies
    phi = np.insert(phi, 0, 0) + pPhase  # Phases - enforce first phase as zero phase
    # phi = [0, phi] + pPhase                                     # Phases - enforce first phase as zero phase

    omega = np.array([beta] * len(t))
    Ft = np.array([[0]] * len(t), dtype='f')
    for i in range(len(t)):
        omega[i] *= t[i]
        omega[i] += phi
        # Ft[i] *= np.cos(omega[i])   # Could be rounding error here
        FtRow = F0 * np.cos(omega[i])  # Could be rounding error here
        Ft[i] += sum(FtRow)

    return x, Ft


def globalShapeFunction(x, lBeam, nElements, L, nDOF, RDOF, Ng, dNg=False, ddNg=False):
    # This function assembles the DOF force matrix based on a time vector and a
    # force vector.The arguments are:
    # INPUT:
    # x    The    load    position[1]
    # Lbeam    The    beam    length[1]
    # nElements    No.of    elements[1]
    # L         Length    of    each    lement(equal)[1]
    # nDOF  Total    no.of    degrees    of    freedom[1]
    # Ng    Shape    function    vector[nDOF, 1]
    # dNg   1    st    derivate    Shape    function    vector[nDOF, 1]
    # ddNg  2nd derivate Shape function vector  [nDOF, 1]

    if 0 <= x <= lBeam:
        s = int(np.fix(x / L) + 1)
        zeta = (x - (s - 1) * L) / L
        if x > nElements:
            s = nElements
            zeta = 1.0

        # dof = []
        # for i in range(2 * s - 2, 2 * s + 2):
        #     dof.append(i)
        dof = np.array([i for i in range(2*s-2, 2*s+2)])

        N1 = 1 - 3 * zeta ** 2 + 2 * zeta ** 3
        N2 = (zeta - 2 * zeta ** 2 + zeta ** 3) * L
        N3 = 3 * zeta ** 2 - 2 * zeta ** 3
        N4 = (-zeta ** 2 + zeta ** 3) * L
        Ng[dof] = np.array([N1, N2, N3, N4], dtype='f')

        # dN1 = -6 * x / L ** 2 + 6 * x ** 2 / L ** 3
        # dN2 = 1 - 4 * x / L + 3 * x ** 2 / L ** 2
        # dN3 = 6 * x / L ** 2 - 6 * x ** 2 / L ** 3
        # dN4 = -2 * x / L + 3 * x ** 2 / L ** 2
        # dNg[dof] = [dN1, dN2, dN3, dN4]

        # ddN1 = -6 / L ** 2 + 12 * x / L ** 3
        # ddN2 = -4 / L + 6 * x / L ** 2
        # ddN3 = 6 / L ** 2 - 12 * x / L ** 3
        # ddN4 = -2 / L + 6 * x / L ** 2
        # ddNg[dof] = [ddN1, ddN2, ddN3, ddN4]

    for i in RDOF:
        Ng[i] = 0
        # dNg[i] = 0
        # ddNg[i] = 0

    return Ng


fe_mf()

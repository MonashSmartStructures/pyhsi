import math
import numpy as np

from matplotlib import pyplot as plt

from scipy.linalg import eig
from scipy.linalg import eigh

from crowd import *
from beam import *


class Solver:

    nSteps = 5000
    g = 9.81
    PedestrianModel = None
    ModelType = None

    def __init__(self, crowd, beam):
        self.crowd = crowd
        self.beam = beam

        self.nBDOF = self.beam.nBDOF                        # Beam-only DOFs
        self.nDOF = self.calcnDOF()                         # Overall DOFs
        self.RDOF = self.beam.RDOF                          # Restrained DOFs

        self.lamda = np.zeros(self.nSteps, 2 * self.nDOF)   # ???

        self.t, self.dT = self.genTimeVector()              # Setup time vector
        self.Mb, self.Cb, self.Kb = self.assembleMCK()      # Assemble mass, stiffness and damping matrices

        self.q = np.zeros(self.nSteps, self.nDOF)           # Initialize displacement matrix
        self.dq = np.zeros(self.nSteps, self.nDOF)          # Initialize velocity matrix
        self.ddq = np.zeros(self.nSteps, self.nDOF)         # Initialize acceleration matrix
        self.F0 = np.zeros(1, self.nDOF)                    # Initialize ??? matrix

    # region Prepare Solver
    def calcnDOF(self):
        if self.PedestrianModel == "Spring Mass Damper":
            return
        return self.nBDOF + self.crowd.numPedestrians

    def genTimeVector(self):
        f = 1/(2*math.pi) * (math.pi/self.beam.length)**2*math.sqrt(self.beam.EI/self.beam.linearMass)
        period = 1/f
        dTMax = 0.02*period     # Stability of newmark

        maxTimeOff = 0
        for ped in self.crowd.pedestrians:
            timeOff = ped.calcTimeOff(self.beam.length)
            if timeOff > maxTimeOff:
                maxTimeOff = timeOff

        timeEnd = 1.1*maxTimeOff    # Run simulation for a bit after the last ped has left
        dT = timeEnd / self.nSteps
        dT = min(dT, dTMax)
        t = np.arange(0, timeEnd, dT)  # Rounding error created by differing precision in Python vs MATLAB

        self.nSteps = len(t)    # Adjust nSteps

        return t, dT

    def assembleMCK(self):
        """
        This function assembles the mass and stiffness matrices, applies the boundary conditions, and calculates a
        damping matrix based on Rayleigh damping
        """

        # M - System mass
        # C - System damping
        # K - System stiffness

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
        Mt, Ct, Kt = self.constraints(RDOF, Mt, Ct, Kt)

        phi, w = self.modal(Mt, Kt)

        alpha, beta = self.rayleighCoeffs(w, self.beam.modalDampingRatio, self.beam.nHigh)

        C = alpha * M + beta * K

        # self.Mb = M
        # self.Cb = C
        # self.Kb = K
        return M, C, K

    @staticmethod
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

    @staticmethod
    def modal(M, K):
        lam, phi = eigh(K, M)
        n, m = K.shape
        omega = np.sqrt(lam)

        return phi, omega

    @staticmethod
    def rayleighCoeffs(w, modalDampingRatio, nHigh):
        wr = [i for i in w if i > 1.01]
        wi = wr[0]
        wj = wr[nHigh - 1]

        alpha = modalDampingRatio * (2 * wi * wj) / (wi + wj)
        beta = modalDampingRatio * 2 / (wi + wj)

        return alpha, beta
    # endregion

    # region Solve
    def solve(self):
        for i in range(1, self.nSteps):
            u, du, ddu, Ft = self.nonLinearNewmarkBeta()
            self.F0 = Ft
            self.q[i:] = u
            self.dq[i:] = du
            self.ddq[i:] = ddu

    def nonLinearNewmarkBeta(self):
        # TODO: Complete
        u = []
        du = []
        ddu = []
        Ft = []
        lamda = self.lamda
        return u, du, ddu, Ft

    def getCurrentSystemMatrices(self, t):
        # This function returns the M, C, K and F matrices at time t

        # Apply constraints to beam
        self.applyConstraints()

        # Initialize global matrices
        M = self.Mb
        C = self.Cb
        K = self.Kb
        F = np.zeros(1, self.nDOF)

        # Shape function zero matrices
        Ng0 = np.zeros(1, self.nBDOF)
        dNg0 = np.zeros(1, self.nBDOF)
        ddNg0 = np.zeros(1, self.nBDOF)
        elementLength = self.beam.length / self.beam.numElements

        # For each pedestrian
        for ped in self.crowd.pedestrians:
            x, Ft = ped.calcPedForce(t)     # Pedestrian position and force
            N, dN, ddN = self.globalShapeFunction(x, Ng0, dNg0, ddNg0)

            # Calculate adjustments to MCKF
            MStar = ped.mass * np.transpose(N) * N
            CStar = ped.mass * ped.veloicty * 2 * np.transpose(N) * dN
            KStar = ped.mass * ped.veloicty ** 2 * np.transpose(N) * ddN
            Fp = N * Ft

            # Assemble into global matrices
            M += MStar
            C += CStar
            K += KStar
            F += Fp

        return M, C, K, F

    def applyConstraints(self):
        def imposeRestraint(A, dof):
            A[dof] = 0  # column
            A[:, dof] = 0  # row
            A[dof, dof] = 1  # diagonal

            return A

        for i in self.RDOF:
            dof = i
            self.Mb = imposeRestraint(self.Mb, dof)
            self.Cb = imposeRestraint(self.Cb, dof)
            self.Kb = imposeRestraint(self.Kb, dof)

    def globalShapeFunction(self, x, Ng, dNg, ddNg):
        """
        This function assembles the DOF force matric based on a time vector and a force vector
        """

        # Check if the force is on the bridge
        if self.beam.onBeam(x):
            L = self.beam.elemLength
            s, zeta = self.beam.locationOnBeam(x)
            dof = np.array([i for i in range(2 * s - 2, 2 * s + 2)])

            N1 = 1 - 3 * zeta ** 2 + 2 * zeta ** 3
            N2 = (zeta - 2 * zeta ** 2 + zeta ** 3) * L
            N3 = 3 * zeta ** 2 - 2 * zeta ** 3
            N4 = (-zeta ** 2 + zeta ** 3) * L
            Ng[dof] = np.array([N1, N2, N3, N4], dtype='f')

            dN1 = -6 * x / L ** 2 + 6 * x ** 2 / L ** 3
            dN2 = 1 - 4 * x / L + 3 * x ** 2 / L ** 2
            dN3 = 6 * x / L ** 2 - 6 * x ** 2 / L ** 3
            dN4 = -2 * x / L + 3 * x ** 2 / L ** 2
            dNg[dof] = [dN1, dN2, dN3, dN4]

            ddN1 = -6 / L ** 2 + 12 * x / L ** 3
            ddN2 = -4 / L + 6 * x / L ** 2
            ddN3 = 6 / L ** 2 - 12 * x / L ** 3
            ddN4 = -2 / L + 6 * x / L ** 2
            ddNg[dof] = [ddN1, ddN2, ddN3, ddN4]

        for i in self.RDOF:
            Ng[i] = 0
            # dNg[i] = 0
            # ddNg[i] = 0

        return Ng, dNg, ddNg
    # endregion


class FeMmSolver(Solver):

    PedestrianModel = "Moving Mass"
    ModelType = "Finite Element"

    # Solve class is based on a FE MM System


class FeMfSolver(Solver):

    PedestrianModel = "Moving Force"
    ModelType = "Finite Element"

    # region Solve
    def getCurrentSystemMatrices(self, t):
        # TODO: Rewrite for FE MF System
        pass
    # endregion


class FeSMDSolver(Solver):

    PedestrianModel = "Spring Mass Damper"
    ModelType = "Finite Element"

    # region Solve
    def getCurrentSystemMatrices(self, t):
        # TODO: Rewrite for FE SMD System
        pass
    # endregion


class MoMmSolver(Solver):

    PedestrianModel = "Moving Mass"
    ModelType = "Modal Analysis"

    # region Solve
    def getCurrentSystemMatrices(self, t):
        # TODO: Rewrite for MO SMD System
        # This function returns the M, C, K and F matrices at time t

        beta = math.pi/self.beam.length * np.array([i for i in range(self.beam.numElements)], dtype='f')
        w = beta**2 * math.sqrt(self.beam.EI/self.beam.linearMass)

        # Initialize global matrices
        M = np.eye(self.beam.numElements)
        C = np.diag(2*self.beam.modalDampingRatio*w)
        K = np.diag(w**2)
        F = np.zeros(1, self.beam.numElements)

        # Shape function zero matrices
        Ng0 = np.zeros(1, self.nBDOF)
        dNg0 = np.zeros(1, self.nBDOF)
        ddNg0 = np.zeros(1, self.nBDOF)
        elementLength = self.beam.length / self.beam.numElements

        # For each pedestrian
        for ped in self.crowd.pedestrians:
            x, Ft = ped.calcPedForce(t)  # Pedestrian position and force
            N, dN, ddN = self.globalShapeFunction(x, Ng0, dNg0, ddNg0)

            # Calculate adjustments to MCKF
            MStar = ped.mass * np.transpose(N) * N
            CStar = ped.mass * ped.veloicty * 2 * np.transpose(N) * dN
            KStar = ped.mass * ped.veloicty ** 2 * np.transpose(N) * ddN
            Fp = N * Ft

            # Assemble into global matrices
            M += MStar
            C += CStar
            K += KStar
            F += Fp

        return M, C, K, F
    # endregion


class MoMfSolver(Solver):

    PedestrianModel = "Moving Force"
    ModelType = "Modal Analysis"

    # region Solve
    def getCurrentSystemMatrices(self, t):
        # TODO: Rewrite for MO SMD System
        pass
    # endregion


class MoSMDSolver(Solver):

    PedestrianModel = "Spring Mass Damper"
    ModelType = "Modal Analysis"

    # region Solve
    def getCurrentSystemMatrices(self, t):
        # TODO: Rewrite for MO SMD System
        pass
    # endregion



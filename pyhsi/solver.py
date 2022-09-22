import math

import numpy
import numpy as np

from matplotlib import pyplot as plt

from scipy.linalg import eig
from scipy.linalg import eigh

from crowd import *
from beam import *
''' import the beams from the beam'''

class Solver:
    '''This section of the code creates a solver class'''
    nSteps = 5000
    g = 9.81
    PedestrianModel = None
    ModelType = None

    def __init__(self, crowd, beam):
        ''' function defines the object properties crowd and beam'''
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
        ''' this function generates the time vector by simulating the time frame for a given space and stride vector'''
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

    def nonLinearNewmarkBeta(t, dt, u0, du0, ddu0, F0, func, eigflag):
        '''This function obtains the matrices M,C,K and F at time t from the function func. It integrates the euquations over time
         step dt and returns accelerations, velocities and displacements. it passes the current values of these parameters to func.'''
        # TODO: Complete
        # u = displacement
        # du = velocity
        # ddu = acceleration
        # Ft = force
        # lamda = ??

        gamma = 1/2
        beta = 1/4
        force_tol = 10**0-6
        max_inc = 20

        #Get current system matrices
        M, C, K, F = func(t, u0, du0, ddu0)
        nDOF = len(F)
        dF = F-F0

        # if eigenvalue output is requested, produce the complex eigenvalues
        lamda = np.zeros(2*nDOF, 1)

        if nargin < 8 or nargout < 5:
            #not really sure where this value comes from
            eigflag = 0

        if eigflag != 0:
            A1 = np.zeros(nDOF)
            A2 = np.array(nDOF)
            A3 = np.linalg.lstsq(-M,K)[0]
            A4 = np.linalg.lstsq(-M,C)[0]
            A = [[A1,A2],[A3,A4]]
            lamda = eig(A)
            lamda = np.transpose(lamda)

        a0 = 1/(beta*dt^2)
        a1 = gamma/(beta*dt)
        a2 = 1/(beta*dt)
        a3 = gamma/beta
        a4 = 1/(2*beta)
        a5 = dt/2*(gamma/beta-2)
        a6 = dt*(1-gamma/(2*beta))

        Keff = a0*M + a1*C + K
        #iKeff = numpy.linalg.inv(Keff)

        A = a2*M + a3*C
        B = a4*M + a5*C

        # Effective force and current displacement step
        dF_trans = numpy.transpose(dF)
        duO_trans = numpy.transpose(du0)
        ddu0_trans = numpy.transpose(ddu0)
        dFeff = dF_trans + A*duO_trans + B*ddu0_trans
        delta_u = np.linalg.lstsq(Keff, dFeff)[0]

        #incremental displacement
        del_u_i = 0
        not_done = true
        i_inc = 0


        while not_done and i_inc < max_inc:
            delta_u = delta_u + del_u_i
            delta_du = a1*delta_u - a3*duO_trans + a6*ddu0_trans
            delta_ddu = a0*delta_u - a2*duO_trans - a4*ddu0_trans

            del_u_trans = numpy.transpose(delta_u)
            del_du_trans = numpy.transpose(delta_du)
            del_ddu_trans = numpy.transpose(delta_ddu)

            u = u0 + del_u_trans
            du = du0 + del_du_trans
            ddu = ddu0 + del_ddu_trans

            u_trans = numpy.transpose(u)
            du_trans = numpy.trans(du)
            ddu_trans = numpy.transpose(ddu)


            Res = F - (M* ddu_trans + C*du_trans +K*u_trans)
            #residual force
            Res_trans = numpy.transpose(Res)
            del_u_i = Keff/Res_trans
            not_done  = numpy.linalg.norm(Res) > force_tol
            i_inc = i_inc + 1


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



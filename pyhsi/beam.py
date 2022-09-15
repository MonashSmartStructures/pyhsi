import math
import numpy as np


class Beam:
    # Class attributes
    numElements = 10  # n - Number of beam elements
    length = 50  # L - Length (m)
    width = 2  # b - Width (m)
    height = 0.6  # h - Height (m)
    E = 200e9  # E - Young's modulus (N/m^2)
    modalDampingRatio = 0.005  # xi - Modal damping ratio of the beam
    nHigh = 3  # nHigh - Higher mode for damping matrix
    area = 0.3162  # A - Cross-section area (m^2)
    linearMass = 500  # m - Linear mass (kg/m)

    beamFreq = 2  # f - Beam frequency, given linear mass (Hz)

    def __init__(self):
        # Allow any beam properties to be passed at argument?

        self.elemLength = None
        self.I = None
        self.EI = None
        self.nDOF = 0
        self.nBDOF = 0
        self.RDOF = 0

        self.calcBeamProperties()

    def calcBeamProperties(self):
        self.I = (self.width * self.height ** 3) / 12  # I - Second Moment of Area (m^4)
        self.EI = self.linearMass * (
                    (2 * math.pi * self.beamFreq) * (math.pi / self.length) ** (-2)) ** 2  # EI - Flexural Rigidity
        self.nDOF = 2 * (self.numElements + 1)
        self.nBDOF = 2 * (self.numElements + 1)
        self.RDOF = [0, self.nDOF - 2]  # Should this be nDOF-1 so that the last column is used not 2nd last?

        if self.numElements % 2 != 0:
            self.numElements += 1

        self.elemLength = self.length/self.numElements

    def testBeamProperties(self):
        return

    def beamElement(self):
        L = self.elemLength

        # Elemental mass matrix
        elementalMassMatrix = np.array([[156, 22 * L, 54, -13 * L], [22 * L, 4 * L ** 2, 13 * L, -3 * L ** 2],
                                        [54, 13 * L, 156, -22 * L], [-13 * L, -3 * L ** 2, -22 * L, 4 * L ** 2]],
                                       dtype='f')
        elementalMassMatrix *= (self.linearMass*L/420)

        # Elemental stiffness matrix
        elementalStiffnessMatrix = np.array([[12, 6 * L, -12, 6 * L], [6 * L, 4 * L ** 2, -6 * L, 2 * L ** 2],
                                             [-12, -6 * L, 12, -6 * L], [6 * L, 2 * L ** 2, -6 * L, 4 * L ** 2]],
                                            dtype='f')
        elementalStiffnessMatrix *= (self.EI/L**3)

        return elementalMassMatrix, elementalStiffnessMatrix

    def onBeam(self, x):
        # Checks if a location is on the beam
        if 0 <= x <= self.length:
            return True
        else:
            return False

    def locationOnBeam(self, x):
        # Returns which element x is on and where on that element it is
        elemNumber = int(np.fix(x / self.elemLength) + 1)
        elemLocation = (x - (elemNumber - 1) * self.elemLength) / self.elemLength
        if elemNumber > self.numElements:
            elemNumber = self.numElements
            elemLocation = 1.0
        return elemNumber, elemLocation

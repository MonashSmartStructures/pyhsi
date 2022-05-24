
import math


class Beam:

    # Class attributes
    numElements = 10            # n - Number of beam elements
    length = 50                 # L - Length (m)
    width = 2                   # b - Width (m)
    height = 0.6                # h - Height (m)
    E = 200e9       # E - Young's modulus (N/m^2)
    modalDampingRatio = 0.005   # xi - Modal damping ratio of the beam
    nHigh = 3                   # nHigh - Higher mode for damping matrix
    area = 0.3162               # A - Cross-section area (m^2)
    linearMass = 500            # m - Linear mass (kg/m)

    beamFreq = 2                # f - Beam frequency, given linear mass (Hz)

    def __init__(self):
        # Allow any beam properties to be passed at argument?

        self.I = None
        self.EI = None
        self.nDOF = None
        self.RDOF = None

        self.calcBeamProperties()

    def calcBeamProperties(self):
        self.I = (self.width * self.height ** 3) / 12  # I - Second Moment of Area (m^4)
        self.EI = self.linearMass * ((2 * math.pi * self.beamFreq) * (math.pi / self.length) ** (-2)) ** 2  # EI - Flexural Rigidity
        self.nDOF = 2 * (self.numElements + 1)
        self.RDOF = [1, self.nDOF-1]

        if self.numElements % 2 != 0:
            self.numElements += 1

    def testBeamProperties(self):

        return

    def beamElement(self):
        L = self.length

        # Elemental mass matrix
        elementalMassMatrix = [[156, 22 * L, 54, -13 * L], [22 * L, 4 * L ** 2, 13 * L, -3 * L ** 2],
                               [54, 13 * L, 156, -22 * L], [-13 * L, -3 * L ** 2, -22 * L, 4 * L ** 2]]

        # Elemental stiffness matrix
        elementalStiffnessMatrix = [[12, 6*L, -12, 6*L], [6*L, 4*L**2, -6*L, 2*L**2],
                                    [-12, -6*L, 12, -6*L], [6*L, 2*L**2, -6*L, 4*L**2]]

        return elementalMassMatrix, elementalStiffnessMatrix

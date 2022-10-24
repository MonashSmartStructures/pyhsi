"""
PyHSI - Beam Class definition
"""
import math
import numpy as np


class Beam:
    """
    Class definition
    """
    # Class attributes
    _numElements = 10  # n - Number of beam elements !not for modal
    length = 50  # L - Length (m)
    width = 2  # b - Width (m)
    height = 0.6  # h - Height (m)
    E = 200e9  # E - Young's modulus (N/m^2)
    modalDampingRatio = 0.005  # xi - Modal damping ratio of the beam
    nHigh = 3  # nHigh - Higher mode for damping matrix
    area = 0.3162  # A - Cross-section area (m^2)
    linearMass = 500  # m - Linear mass (kg/m)

    beamFreq = 2  # f - Beam frequency, given linear mass (Hz)

    def __init__(self, numElements, length, width, height, E, modalDampingRatio, nHigh, area, linearMass):
        """
        Constructs a beam object

        Parameters
        ----------
        numElements: int
            The number of beam element
        length: int
            Length of beam
        width: int
            Width of beam
        height: int
            Height of beam
        E: Any
            Young's modulus
        modalDampingRatio: Any
            Modal damping ratio of the beam
        nHigh: Any
            Higher mode for damping matrix
        area: Any
            Cross-section area
        linearMass: Any
            Linear Mass

        Returns
        -------
        None.
        """

        self._numElements = numElements
        self.length = length
        self.width = width
        self.height = height
        self.E = E
        self.modalDampingRatio = modalDampingRatio
        self.nHigh = nHigh
        self.area = area
        self.linearMass = linearMass

        self._elemLength = None
        self._I = None
        self._EI = None
        self._nDOF = None
        self._nBDOF = None
        self._RDOF = None

    # region Properties
    @property
    def elemLength(self):
        if self._elemLength is None:
            self._elemLength = self.length / self.numElements
        return self._elemLength

    @property
    def I(self):
        # I - Second Moment of Area (m^4)
        if self._I is None:
            self._I = (self.width * self.height ** 3) / 12
        return self._I

    @property
    def EI(self):
        # EI - Flexural Rigidity
        if self._EI is None:
            self._EI = self.linearMass * ((2 * math.pi * self.beamFreq) * (math.pi / self.length) ** (-2)) ** 2
        return self._EI

    @property
    def nDOF(self):
        if self._nDOF is None:
            self._nDOF = 2 * (self.numElements + 1)
        return self._nDOF

    @property
    def nBDOF(self):
        if self._nBDOF is None:
            self._nBDOF = 2 * (self.numElements + 1)
        return self._nBDOF

    @property
    def RDOF(self):
        if self._RDOF is None:
            self._RDOF = [0, self.nDOF - 2]  # Should this be nDOF-1 so that the last column is used not 2nd last?
        return self._RDOF

    @property
    def numElements(self):
        return self._numElements

    @numElements.setter
    def numElements(self, numElements):
        if numElements % 2 != 0:
            numElements += 1
        self._numElements = numElements
        self._elemLength = None
    # endregion

    def beamElement(self):
        L = self.elemLength

        # Elemental mass matrix
        elementalMassMatrix = np.array([[156, 22 * L, 54, -13 * L], [22 * L, 4 * L ** 2, 13 * L, -3 * L ** 2],
                                        [54, 13 * L, 156, -22 * L], [-13 * L, -3 * L ** 2, -22 * L, 4 * L ** 2]],
                                       dtype='f')
        elementalMassMatrix *= (self.linearMass * L / 420)

        # Elemental stiffness matrix
        elementalStiffnessMatrix = np.array([[12, 6 * L, -12, 6 * L], [6 * L, 4 * L ** 2, -6 * L, 2 * L ** 2],
                                             [-12, -6 * L, 12, -6 * L], [6 * L, 2 * L ** 2, -6 * L, 4 * L ** 2]],
                                            dtype='f')
        elementalStiffnessMatrix *= (self.EI / L ** 3)

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

    @classmethod
    def fromDict(cls, beamProperties):
        numElements = beamProperties['numElements']
        length = beamProperties['length']
        width = beamProperties['width']
        height = beamProperties['height']
        E = beamProperties['E']
        modalDampingRatio = beamProperties['modalDampingRatio']
        nHigh = beamProperties['nHigh']
        area = beamProperties['area']
        linearMass = beamProperties['linearMass']

        return cls(numElements, length, width, height, E, modalDampingRatio, nHigh, area, linearMass)


class FeBeam(Beam):
    pass


class MoBeam(Beam):
    pass

# numElements
# length
# width
# height
# E
# modalDampingRatio
# nHigh
# area
# linearMass

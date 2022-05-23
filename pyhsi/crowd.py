import numpy as np
import math
import matplotlib.pyplot as plt

class Crowd:

    # Class attributes - crowd_props in matlab
    # meanMass = 74.1             # mM
    # sdMass = 15.91              # sM
    meanLognormalModel = 4.28   # mM
    sdLognormalModel = 0.21     # sM
    meanPace = 1.96             # mP
    sdPace = 0.209              # sP
    meanStride = 0.66           # mS
    sdStride = 0.066            # sS
    meanStiffness = 28000       # mK
    sdStiffness = 2800          # sK
    meanDamping = 0.3           # mXi
    sdDamping = 0.03            # sXi

    def __init__(self, density, length, width, sync):
        self.density = density  # ro
        self.length = length    # Lc
        self.width = width      # Wc
        self.sync = sync        # Sc

        self.area = self.length * self.width                # Ac
        self.avgNumInCrowd = int(self.density * self.area)  # Nc
        self.lamda = self.avgNumInCrowd / self.length       # Avg person/m

    def generateLocations(self, lamda, avgNumCrowd):
        self.gaps = np.random.exponential(1/self.lamda, size=self.avgNumInCrowd)
        self.pLoc = np.cumsum(self.gaps, axis=None, dtype=None, out=None)

    def generateBodyProperties(self):
        self.pMass = np.random.lognormal(mean=self.meanLognormalModel, sigma=self.sdLognormalModel, size=self.avgNumInCrowd)  # lognrnd(mM,sM,[1 Nc]) log - normal distribution of mass
        self.pXi = np.random.normal(loc=self.meanDamping, scale=self.sdDamping, size=self.avgNumInCrowd) #normrnd(mXi,sXi,[1 Nc])
        self.pStiff = np.random.normal(loc=self.meanStiffness, scale=self.sdStiffness, size=self.avgNumInCrowd) #normrnd(mK, sK, [1 Nc])
        self.pPace = np.random.normal(self.meanPace,self.sdPace,self.avgNumInCrowd)#normrnd(mP, sP, [1 Nc])
        self.pStride = np.random.normal(self.meanStride,self.sdStride,self.avgNumInCrowd)#normrnd(mS, sS, [1 Nc])
        self.pPhase = (2 * math.pi) * np.random.rand(self.avgNumInCrowd)

    def assembleCrowd(self):
        return

c1 = Crowd(3,10,4,3)
c1.generateBodyProperties()
print(c1.pStride)
import numpy as np
import math
import csv
import matplotlib.pyplot as plt


# In the future, implement crowd as a group of pedestrian objects

class oldCrowd:
    # Class attributes - crowd_props in matlab
    # meanMass = 74.1             # mM
    # sdMass = 15.91              # sM
    meanLognormalModel = 4.28  # mM
    sdLognormalModel = 0.21  # sM
    meanPace = 1.96  # mP
    sdPace = 0.209  # sP
    meanStride = 0.66  # mS
    sdStride = 0.066  # sS
    meanStiffness = 28000  # mK
    sdStiffness = 2800  # sK
    meanDamping = 0.3  # mXi
    sdDamping = 0.03  # sXi

    def __init__(self, density, length, width, sync):
        self.density = density  # ro
        self.length = length  # Lc
        self.width = width  # Wc
        self.sync = sync  # Sc

        self.area = self.length * self.width  # Ac
        self.avgNumInCrowd = int(self.density * self.area)  # Nc
        self.lamda = self.avgNumInCrowd / self.length  # Avg person/m

    def generateLocations(self, lamda, avgNumCrowd):
        self.gaps = np.random.exponential(1 / self.lamda, size=self.avgNumInCrowd)
        self.pLoc = np.cumsum(self.gaps, axis=None, dtype=None, out=None)

    def generateBodyProperties(self):
        self.pMass = np.random.lognormal(mean=self.meanLognormalModel, sigma=self.sdLognormalModel,
                                         size=self.avgNumInCrowd)  # lognrnd(mM,sM,[1 Nc]) log - normal distribution of mass
        self.pXi = np.random.normal(loc=self.meanDamping, scale=self.sdDamping,
                                    size=self.avgNumInCrowd)  # normrnd(mXi,sXi,[1 Nc])
        self.pStiff = np.random.normal(loc=self.meanStiffness, scale=self.sdStiffness,
                                       size=self.avgNumInCrowd)  # normrnd(mK, sK, [1 Nc])
        self.pPace = np.random.normal(self.meanPace, self.sdPace, self.avgNumInCrowd)  # normrnd(mP, sP, [1 Nc])
        self.pStride = np.random.normal(self.meanStride, self.sdStride, self.avgNumInCrowd)  # normrnd(mS, sS, [1 Nc])
        self.pPhase = (2 * math.pi) * np.random.rand(self.avgNumInCrowd)

        if self.sync > 0 and self.sync <= 1:
            self.Ns = np.fix(self.avgNumInCrowd * self.sync)
            # Randomly choose the synchronised pedestrians
            self.rp = np.random.permutation(self.avgNumInCrowd)  # randomise indices
            # self.iSync = self.rp[1:self.Ns].sort                #choose first Ns and sort
            # Make the pacing frequencies & phase the same, but random
            self.sPace = np.random.normal(self.meanPace, self.sdPace, size=1)
            # self.pPace[self.iSync] = self.sPace
            self.sPhase = (2 * math.pi) * (np.random.rand(1))
            # self.pPhase[self.iSync] = self.sPhase
        # self.pW = math.sqrt(np.divide(self.pStiff,self.pMass))  #natural frequency of the sprung mass
        # self.pDamp = 2 * np.multiply(self.pMass,self.pW,self.pXi)   #damping coefficient of the sprung mass
        self.pVel = np.multiply(self.pPace, self.pStride)  # Velocity

    def assembleCrowd(self):
        return


class TestCrowd:
    def __init__(self, pMass, pDamp, pStiff, pPace, pPhase, pLoc, pVel, iSync):
        self.pMass = pMass
        self.pDamp = pDamp
        self.pStiff = pStiff
        self.pPace = pPace
        self.pPhase = pPhase
        self.pLoc = pLoc
        self.pVel = pVel
        self.iSync = iSync

        self.size = 1  # The number of pedestrians is 1


class Pedestrian:

    humanProperties = {}

    def __init__(self, pMass, pDamp, pStiff, pPace, pPhase, pLoc, pVel, iSync):
        self.pMass = pMass
        self.pDamp = pDamp
        self.pStiff = pStiff
        self.pPace = pPace
        self.pPhase = pPhase
        self.pLoc = pLoc
        self.pVel = pVel
        self.iSync = iSync

    @classmethod
    def setHumanProperties(cls, humanProperties):
        cls.humanProperties = humanProperties

    @classmethod
    def deterministicPedestrian(cls):
        pMass = cls.humanProperties['meanMass']
        pDamp = 0
        pStiff = 0
        pPace = 0
        pPhase = 0
        pLoc = 0
        pVel = 0
        iSync = 0
        return cls(pMass, pDamp, pStiff, pPace, pPhase, pLoc, pVel, iSync)

    @classmethod
    def randomPedestrain(cls):
        pMass = 0
        pDamp = 0
        pStiff = 0
        pPace = 0
        pPhase = 0
        pLoc = 0
        pVel = 0
        iSync = 0
        return cls(pMass, pDamp, pStiff, pPace, pPhase, pLoc, pVel, iSync)


class Crowd:
    def __init__(self, nPed, density, length, width, sync):
        self.nPed = nPed
        self.density = density
        self.length = length
        self.width = width
        self.sync = sync

        self.pedestrians = []

    def addRandomPedestrian(self):
        self.pedestrians.append(Pedestrian.randomPedestrain())

    def addDeterministicPedestrian(self):
        self.pedestrians.append(Pedestrian.deterministicPedestrian())


class SinglePedestrian(Pedestrian):
    pass


class DeterministicCrowd(Crowd):
    def __init__(self, nPed, density, length, width, sync):
        super().__init__(nPed, density, length, width, sync)

    def populateCrowd(self):
        for i in range(self.nPed):
            self.addDeterministicPedestrian()


class RandomCrowd(Crowd):
    def __init__(self, nPed, density, length, width, sync):
        super().__init__(nPed, density, length, width, sync)

    def populateCrowd(self):
        for i in range(self.nPed):
            self.addRandomPedestrian()


def getHumanProperties():
    humanProperties = {}

    with open('HumanProperties.csv', newline='') as csvFile:
        csvReader = csv.reader(csvFile, delimiter=',')
        lineCount = 0
        for row in csvReader:
            if lineCount > 0:
                humanProperties['mean' + row[0]] = float(row[1])
                humanProperties['sd' + row[0]] = float(row[2])
                # print(f'{row[0]} has mean {row[1]} and standard deviation {row[2]}.')
            lineCount += 1

    return humanProperties

# testcrowd = Crowd(0.5,100,2,0.1)
# testcrowd.generateBodyProperties()
# print(testcrowd.sPhase)

# testcrowd = testCrowd(80,650,21500,2.10,math.pi,0,1.51,0)




import numpy as np
import math
import csv


class Pedestrian:

    humanProperties = {}
    meanLognormalModel = 4.28  # mM
    sdLognormalModel = 0.21  # sM

    detK = 14110
    detVelocity = 1.25

    synchedPace = 0
    synchedPhase = 0

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
    def setPaceAndPhase(cls, pace, phase):
        cls.pace = pace
        cls.phase = phase

    @classmethod
    def deterministicPedestrian(cls, location, synched=0):
        hp = cls.humanProperties
        pMass = hp['meanMass']
        pDamp = hp['meanDamping']*2*math.sqrt(cls.detK*hp['meanMass'])
        pStiff = cls.detK
        pLoc = location

        if synched == 1:
            iSync = 1
            pPace = cls.synchedPace
            pPhase = cls.synchedPhase
        else:
            iSync = 0
            pPace = np.random.normal(hp['meanPace'], hp['sdPace'])
            pPhase = (2 * math.pi) * np.random.rand()

        pVel = cls.detVelocity

        return cls(pMass, pDamp, pStiff, pPace, pPhase, pLoc, pVel, iSync)

    @classmethod
    def randomPedestrian(cls, location, synched=0):
        hp = cls.humanProperties
        pMass = np.random.lognormal(mean=cls.meanLognormalModel, sigma=cls.sdLognormalModel)
        pDamp = np.random.normal(loc=hp['meanDamping'], scale=hp['sdDamping'])
        pStiff = np.random.normal(loc=hp['meanStiffness'], scale=hp['sdStiffness'])
        pLoc = location

        if synched == 1:
            iSync = 1
            pPace = cls.synchedPace
            pPhase = cls.synchedPhase
        else:
            iSync = 0
            pPace = np.random.normal(hp['meanPace'], hp['sdPace'])
            pPhase = (2 * math.pi) * np.random.rand()

        pStride = np.random.normal(hp['meanStride'], hp['sdStride'])
        pVel = np.multiply(pPace, pStride)

        return cls(pMass, pDamp, pStiff, pPace, pPhase, pLoc, pVel, iSync)


class Crowd:

    humanProperties = {}

    def __init__(self, numPedestrians, length, width, sync):
        # self.density = density
        self.numPedestrians = numPedestrians
        self.length = length
        self.width = width
        self.sync = sync

        self.area = self.length * self.width
        # self.numPedestrians = int(self.density * self.area)
        self.lamda = self.numPedestrians / self.length

        self.locations = []
        self.iSync = []
        self.pedestrians = []

        # Crowd synchronization
        self.determineCrowdSynchronisation()

    def determineCrowdSynchronisation(self):
        self.iSync = np.random.choice([0, 1], size=self.numPedestrians, p=[1 - self.sync, self.sync])
        pace = np.random.normal(loc=self.humanProperties['meanPace'], scale=self.humanProperties['sdPace'])
        phase = (2 * math.pi) * (np.random.rand())
        Pedestrian.setPaceAndPhase(pace, phase)

    def addRandomPedestrian(self, location, synched):
        self.pedestrians.append(Pedestrian.randomPedestrian(location, synched))

    def addDeterministicPedestrian(self, location, synched):
        self.pedestrians.append(Pedestrian.deterministicPedestrian(location, synched))

    @classmethod
    def setHumanProperties(cls, humanProperties):
        cls.humanProperties = humanProperties


class SinglePedestrian(Pedestrian):

    def __init__(self):
        k = 14.11e3

        pMass = self.humanProperties['meanMass']
        pDamp = self.humanProperties['meanDamping'] * 2 * math.sqrt(k * pMass)
        pStiff = k
        pPace = 2
        pPhase = 0
        pLoc = 0
        pVel = 1.25
        iSync = 0
        super().__init__(pMass, pDamp, pStiff, pPace, pPhase, pLoc, pVel, iSync)

    pass


class DeterministicCrowd(Crowd):

    arrivalGap = 1      # HSI Paper Section 5.4

    def __init__(self, density, length, width, sync):
        super().__init__(density, length, width, sync)
        self.generateLocations()
        self.populateCrowd()

    def generateLocations(self):
        self.locations = -self.arrivalGap*np.array(range(self.numPedestrians))

    def populateCrowd(self):
        for i in range(self.numPedestrians):
            self.addDeterministicPedestrian(self.locations[i], self.iSync[i])


class RandomCrowd(Crowd):
    def __init__(self, density, length, width, sync):
        super().__init__(density, length, width, sync)
        self.generateLocations()
        self.populateCrowd()

    def generateLocations(self):
        gaps = np.random.exponential(1 / self.lamda, size=self.numPedestrians)
        self.locations = np.cumsum(gaps, axis=None, dtype=None, out=None)

    def populateCrowd(self):
        for i in range(self.numPedestrians):
            self.addRandomPedestrian(self.locations[i], self.iSync[i])


def getHumanProperties():
    humanProperties = {}

    with open('../pyhsi/HumanProperties.csv', newline='') as csvFile:
        csvReader = csv.reader(csvFile, delimiter=',')
        lineCount = 0
        for row in csvReader:
            if lineCount > 0:
                humanProperties['mean' + row[0]] = float(row[1])
                humanProperties['sd' + row[0]] = float(row[2])
            lineCount += 1

    return humanProperties


def updateHumanProperties(humanProperties):
    Pedestrian.setHumanProperties(humanProperties)
    Crowd.setHumanProperties(humanProperties)


# testcrowd = Crowd(0.5,100,2,0.1)
# testcrowd.generateBodyProperties()

# testcrowd = testCrowd(80,650,21500,2.10,math.pi,0,1.51,0)




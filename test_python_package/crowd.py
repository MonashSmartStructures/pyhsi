
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
        return

    def generateBodyProperties(self):
        return

    def assembleCrowd(self):
        return
    
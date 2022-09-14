import pyhsi as hsi

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def testRandomCrowd():
    # Checks the distribution of parameters for random crowds
    n = 100     # number of crowds to test over

    humanProperties = hsi.getHumanProperties()
    hsi.updateHumanProperties(humanProperties)

    # Crowd parameters
    numPedestrians = 100
    length = 50
    width = 1
    sync = 0

    # Create n crowds
    crowds = []
    for i in range(n):
        crowds.append(hsi.RandomCrowd(numPedestrians, length, width, sync))

    pAll = getPedestrianDistribution(crowds)
    plotPedestrianDistribution(pAll, 'Distribution of pedestrian properties for random crowd')


def testDeterministicCrowd():
    # Checks the distribution of parameters for random crowds
    n = 100     # number of crowds to test over

    humanProperties = hsi.getHumanProperties()
    hsi.updateHumanProperties(humanProperties)

    # Crowd parameters
    numPedestrians = 100
    length = 50
    width = 1
    sync = 0

    # Create n crowds
    crowds = []
    for i in range(n):
        crowds.append(hsi.DeterministicCrowd(numPedestrians, length, width, sync))

    pAll = getPedestrianDistribution(crowds)
    plotPedestrianDistribution(pAll, 'Distribution of pedestrian properties for deterministic crowd')


def getPedestrianDistribution(crowds):

    # Save arrays of the pedestrian properties
    pAll = {'pMassAll': [], 'pDampAll': [], 'pStiffAll': [], 'pPaceAll': [], 'pPhaseAll': [], 'pLocAll': [],
            'pVelAll': [], 'iSyncAll': []}

    for crowd in crowds:
        for ped in crowd.pedestrians:
            pAll['pMassAll'].append(ped.pMass)
            pAll['pDampAll'].append(ped.pDamp)
            pAll['pStiffAll'].append(ped.pStiff)
            pAll['pPaceAll'].append(ped.pPace)
            pAll['pPhaseAll'].append(ped.pPhase)
            pAll['pLocAll'].append(ped.pLoc)
            pAll['pVelAll'].append(ped.pVel)
            pAll['iSyncAll'].append(ped.iSync)

    return pAll


def plotPedestrianDistribution(pAll, title):

    # Create histogram
    nBins = 50
    fig, axis = plt.subplots(4, 2, figsize=(12, 8), tight_layout=True)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.9)

    # Mass
    ax1 = axis[0, 0]
    data = pAll['pMassAll']
    mean = average(data)
    meanString = 'Mean mass: {:.2f}'.format(mean)
    ax1.hist(data, bins=nBins, weights=np.ones(len(data)) / len(data))
    ax1.set_title("Mass Distribution")
    ax1.axvline(mean, color='k', linestyle='dashed', linewidth=1)
    ax1.text(0.98, 0.85, meanString, horizontalalignment='right', verticalalignment='center', transform=ax1.transAxes, bbox=props)
    print(meanString)

    # Damping
    ax2 = axis[1, 0]
    data = pAll['pDampAll']
    mean = average(data)
    meanString = 'Mean damping: {:.2f}'.format(mean)
    ax2.hist(data, bins=nBins, weights=np.ones(len(data)) / len(data))
    ax2.set_title("Damping Distribution")
    ax2.axvline(mean, color='k', linestyle='dashed', linewidth=1)
    ax2.text(0.98, 0.85, meanString, horizontalalignment='right', verticalalignment='center', transform=ax2.transAxes, bbox=props)
    print(meanString)

    # Stiffness
    ax3 = axis[2, 0]
    data = pAll['pStiffAll']
    mean = average(data)
    meanString = 'Mean stiffness: {:.2f}'.format(mean)
    ax3.hist(data, bins=nBins, weights=np.ones(len(data)) / len(data))
    ax3.set_title("Stiffness Distribution")
    ax3.axvline(mean, color='k', linestyle='dashed', linewidth=1)
    ax3.text(0.98, 0.85, meanString, horizontalalignment='right', verticalalignment='center', transform=ax3.transAxes, bbox=props)
    print(meanString)

    # Pace
    ax4 = axis[3, 0]
    data = pAll['pPaceAll']
    mean = average(data)
    meanString = 'Mean pace: {:.2f}'.format(mean)
    ax4.hist(data, bins=nBins, weights=np.ones(len(data)) / len(data))
    ax4.set_title("Pace Distribution")
    ax4.axvline(mean, color='k', linestyle='dashed', linewidth=1)
    ax4.text(0.98, 0.85, meanString, horizontalalignment='right', verticalalignment='center', transform=ax4.transAxes, bbox=props)
    print(meanString)

    # Phase
    ax5 = axis[0, 1]
    data = pAll['pPhaseAll']
    mean = average(data)
    meanString = 'Mean phase: {:.2f}'.format(mean)
    ax5.hist(data, bins=nBins, weights=np.ones(len(data)) / len(data))
    ax5.set_title("Phase Distribution")
    ax5.axvline(mean, color='k', linestyle='dashed', linewidth=1)
    ax5.text(0.98, 0.85, meanString, horizontalalignment='right', verticalalignment='center', transform=ax5.transAxes, bbox=props)
    print(meanString)

    # Location
    ax6 = axis[1, 1]
    data = pAll['pLocAll']
    mean = average(data)
    meanString = 'Mean location: {:.2f}'.format(mean)
    ax6.hist(data, bins=nBins, weights=np.ones(len(data)) / len(data))
    ax6.set_title("Location Distribution")
    ax6.axvline(mean, color='k', linestyle='dashed', linewidth=1)
    ax6.text(0.98, 0.85, meanString, horizontalalignment='right', verticalalignment='center', transform=ax6.transAxes, bbox=props)
    print(meanString)

    # pVelAll
    ax7 = axis[2, 1]
    data = pAll['pVelAll']
    mean = average(data)
    meanString = 'Mean velocity: {:.2f}'.format(mean)
    ax7.hist(data, bins=nBins, weights=np.ones(len(data)) / len(data))
    ax7.set_title("Velocity Distribution")
    ax7.axvline(mean, color='k', linestyle='dashed', linewidth=1)
    ax7.text(0.98, 0.85, meanString, horizontalalignment='right', verticalalignment='center', transform=ax7.transAxes, bbox=props)
    print(meanString)

    # iSyncAll
    ax8 = axis[3, 1]
    data = pAll['iSyncAll']
    meanString = 'Mean synchronisation: {:.2f}'.format(mean)
    ax8.hist(data, bins=nBins, weights=np.ones(len(data)) / len(data))
    ax8.set_title("Synchronisation Distribution")
    print(meanString)

    # Format plot
    fig.suptitle(title, fontsize=16)
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))

    # Show plot
    plt.show()


def average(lst):
    return sum(lst) / len(lst)


if __name__ == '__main__':
    testRandomCrowd()
    testDeterministicCrowd()

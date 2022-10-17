import pyhsi as hsi

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def testRandomCrowd():
    # Checks the distribution of parameters for random crowds
    n = 100     # number of crowds to test over

    populationProperties = hsi.getPopulationProperties()
    hsi.updatePopulationProperties(populationProperties)

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

    populationProperties = hsi.getPopulationProperties()
    hsi.updatePopulationProperties(populationProperties)

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
    pAll = {'pMassAll': [], 'pDampAll': [], 'pStiffAll': [], 'pPaceAll': [], 'pPhaseAll': [], 'pLocationAll': [],
            'pVelocityAll': [], 'iSyncAll': []}

    for crowd in crowds:
        for ped in crowd.pedestrians:
            pAll['pMassAll'].append(ped.mass)
            pAll['pDampAll'].append(ped.damp)
            pAll['pStiffAll'].append(ped.stiff)
            pAll['pPaceAll'].append(ped.pace)
            pAll['pPhaseAll'].append(ped.phase)
            pAll['pLocationAll'].append(ped.location)
            pAll['pVelocityAll'].append(ped.velocity)
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
    meanString = f'Mean mass: {mean:.2f}'
    ax1.hist(data, bins=nBins, weights=np.ones(len(data)) / len(data))
    ax1.set_title("Mass Distribution")
    ax1.axvline(mean, color='k', linestyle='dashed', linewidth=1)
    ax1.text(0.98, 0.85, meanString, horizontalalignment='right', verticalalignment='center', transform=ax1.transAxes, bbox=props)
    print(meanString)

    # Damping
    ax2 = axis[1, 0]
    data = pAll['pDampAll']
    mean = average(data)
    meanString = f'Mean damping: {mean:.2f}'
    ax2.hist(data, bins=nBins, weights=np.ones(len(data)) / len(data))
    ax2.set_title("Damping Distribution")
    ax2.axvline(mean, color='k', linestyle='dashed', linewidth=1)
    ax2.text(0.98, 0.85, meanString, horizontalalignment='right', verticalalignment='center', transform=ax2.transAxes, bbox=props)
    print(meanString)

    # Stiffness
    ax3 = axis[2, 0]
    data = pAll['pStiffAll']
    mean = average(data)
    meanString = f'Mean stiffness: {mean:.2f}'
    ax3.hist(data, bins=nBins, weights=np.ones(len(data)) / len(data))
    ax3.set_title("Stiffness Distribution")
    ax3.axvline(mean, color='k', linestyle='dashed', linewidth=1)
    ax3.text(0.98, 0.85, meanString, horizontalalignment='right', verticalalignment='center', transform=ax3.transAxes, bbox=props)
    print(meanString)

    # Pace
    ax4 = axis[3, 0]
    data = pAll['pPaceAll']
    mean = average(data)
    meanString = f'Mean pace: {mean:.2f}'
    ax4.hist(data, bins=nBins, weights=np.ones(len(data)) / len(data))
    ax4.set_title("Pace Distribution")
    ax4.axvline(mean, color='k', linestyle='dashed', linewidth=1)
    ax4.text(0.98, 0.85, meanString, horizontalalignment='right', verticalalignment='center', transform=ax4.transAxes, bbox=props)
    print(meanString)

    # Phase
    ax5 = axis[0, 1]
    data = pAll['pPhaseAll']
    mean = average(data)
    meanString = f'Mean phase: {mean:.2f}'
    ax5.hist(data, bins=nBins, weights=np.ones(len(data)) / len(data))
    ax5.set_title("Phase Distribution")
    ax5.axvline(mean, color='k', linestyle='dashed', linewidth=1)
    ax5.text(0.98, 0.85, meanString, horizontalalignment='right', verticalalignment='center', transform=ax5.transAxes, bbox=props)
    print(meanString)

    # Location
    ax6 = axis[1, 1]
    data = pAll['pLocationAll']
    mean = average(data)
    meanString = f'Mean location: {mean:.2f}'
    ax6.hist(data, bins=nBins, weights=np.ones(len(data)) / len(data))
    ax6.set_title("Location Distribution")
    ax6.axvline(mean, color='k', linestyle='dashed', linewidth=1)
    ax6.text(0.98, 0.85, meanString, horizontalalignment='right', verticalalignment='center', transform=ax6.transAxes, bbox=props)
    print(meanString)

    # pVelocityAll
    ax7 = axis[2, 1]
    data = pAll['pVelocityAll']
    mean = average(data)
    meanString = f'Mean velocity: {mean:.2f}'
    ax7.hist(data, bins=nBins, weights=np.ones(len(data)) / len(data))
    ax7.set_title("Velocity Distribution")
    ax7.axvline(mean, color='k', linestyle='dashed', linewidth=1)
    ax7.text(0.98, 0.85, meanString, horizontalalignment='right', verticalalignment='center', transform=ax7.transAxes, bbox=props)
    print(meanString)

    # iSyncAll
    ax8 = axis[3, 1]
    data = pAll['iSyncAll']
    meanString = f'Mean synchronisation: {mean:.2f}'
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

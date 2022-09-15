import numpy as np
import math
import matplotlib.pyplot as plt
from beam import *


def main():

    beam = Beam()
    x = 49
    if beam.onBeam(x):
        elemNumber, elemLocation = beam.locationOnBeam(x)
        print("Element Number: {}\nElement Location: {}".format(elemNumber, elemLocation))


main()

import numpy as np
import math
import matplotlib.pyplot as plt
from beam import *
import numpy as np

def main():

    beam = Beam()
    x = 49
    if beam.onBeam(x):
        elemNumber, elemLocation = beam.locationOnBeam(x)
        print("Element Number: {}\nElement Location: {}".format(elemNumber, elemLocation))


A = [[1, 4, 5, 12],
    [-5, 8, 9, 0],
    [-6, 7, 11, 19]]

print("A =", A)
print("A[1] =", A[1])      # 2nd row
print("A[1][2] =", A[1][2])   # 3rd element of 2nd row
print("A[0][-1] =", A[0][-1])   # Last element of 1st Row

column = [];        # empty list
for row in A:
  column.append(row[2])

print("3rd column =", column)
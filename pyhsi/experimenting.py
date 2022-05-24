import numpy as np


def main():
    print("Experimenting...\n")

    largeArray = np.array([[0]*10]*11)
    smallArray = np.array([[1, 2, 3, 4], [12, 3, 4, 5], [13, 4, 5, 6], [14, 5, 6, 7]])

    print(largeArray)
    print(smallArray)

    # largeArray[0:4, 0:4] = largeArray[0:4, 0:4] + smallArray
    largeArray[0:4, 0:4] += smallArray
    print(largeArray)

    print(largeArray[0])
    print(largeArray[:, 0] + 10)

    row, col = largeArray.shape
    print(row)
    print(col)

main()

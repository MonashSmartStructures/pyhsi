import numpy as np


def main():
    print("Experimenting...\n")

    F = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print(np.diff(F, axis=0))
    print(np.diff(F, 1, 0))


main()

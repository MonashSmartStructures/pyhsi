import numpy as np
import math
import matplotlib.pyplot as plt
from beam import *
import scipy.linalg as la


#def main():
 #   a = 23 // 2
  #  print(a)

#if __name__ == "__main__":
 #   main()


class Employee:
    def __init__(self, first, last, pay): #we also have email but the email can be created with the first name and the last name
        self.first = first
        self.last = last
        self.pay = pay
        self.email = first + '.' + last +'@company.com'


emp_1 = Employee()
emp_2 = Employee()

print(emp_1)
print(emp_2)


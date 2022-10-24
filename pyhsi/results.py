"""
PyHSI - Human Structure Interaction -  Results Module
"""
import xlsxwriter
import inquirer
import openpyxl
import pandas as pd
import math
import numpy as np
import sys
from matplotlib import pyplot as plt

from tkinter import *
from tkinter import filedialog


def loadResults():

    results = Results.open()
    # results = Results.open('../simulations/results/simple_fe_mm.xlsx')
    # results = Results.open('../simulations/results/crowd1_fe_mm.xlsx')

    results.options()


class Results:

    abbr = {
        'Finite Element': 'fe',
        'Modal Analysis': 'mo',
        'Moving Mass': 'mm',
        'Moving Force': 'mf',
        'Spring Mass Damper': 'smd',
    }

    def __init__(self, t, displacement, velocity, acceleration, pedestrianModel=None, modelType=None, filename=None):

        self.t = t
        self.displacement = displacement
        self.velocity = velocity
        self.acceleration = acceleration

        self.pedestrianModel = pedestrianModel
        self.modelType = modelType
        self.filename = filename

        self._midspanAcceleration = None
        self._midspanRMS = None
        self._maxMidspanRMS = None

    # region Properties
    @property
    def midspanAcceleration(self):
        if self._midspanAcceleration is None:
            midspanX = self.acceleration.shape[1] // 2 - 1
            self._midspanAcceleration = self.acceleration[:, midspanX]
        return self._midspanAcceleration

    @property
    def midspanRMS(self):
        if self._midspanRMS is None:
            midspanX = self.acceleration.shape[1] // 2 - 1
            self._midspanRMS = self.calculateRMS(midspanX)
        return self._midspanRMS

    @property
    def maxMidspanRMS(self):
        if self._maxMidspanRMS is None:
            self._maxMidspanRMS = max(self.midspanRMS)
        return self._maxMidspanRMS
    # endregion

    # region Open and Save results
    def askSave(self):
        # Check if user wants to save the results
        saveMessage = "Do you want to save the results?"
        saveChoices = ['Yes', 'No']
        saveQuestion = [inquirer.List('save', message=saveMessage, choices=saveChoices)]
        answer = inquirer.prompt(saveQuestion)

        if answer['save'] == 'Yes':
            self.save()

    def save(self):
        # Get name to save workbook
        filenameMessage = "Enter a filename to save the results under"
        filenameDefault = f"{self.filename[15:-4]}_{self.abbr[self.modelType]}_{self.abbr[self.pedestrianModel]}"
        filenameQuestion = [inquirer.Text('filename', message=filenameMessage, default=filenameDefault)]
        filenameAnswer = inquirer.prompt(filenameQuestion)
        path = f"../simulations/results/{filenameAnswer['filename']}"
        if not path[-5:] == '.xlsx':
            path += '.xlsx'
        self.filename = path

        # Create a Pandas dataframe from the data
        tDF = pd.DataFrame(self.t)
        displacementDF = pd.DataFrame(self.displacement)
        velocityDF = pd.DataFrame(self.velocity)
        accelerationDF = pd.DataFrame(self.acceleration)

        # Create a Panas Excel writer using XlsxWriter as the engine
        writer = pd.ExcelWriter(path, engine='xlsxwriter')

        # Convert the dataframe to an XlsxWriter Excel object
        tDF.to_excel(writer, sheet_name='time', header=False, index=False)
        displacementDF.to_excel(writer, sheet_name='displacement', header=False, index=False)
        velocityDF.to_excel(writer, sheet_name='velocity', header=False, index=False)
        accelerationDF.to_excel(writer, sheet_name='acceleration', header=False, index=False)

        # Close the Pandas Excel writer and output the Excel file
        writer.save()

        print("Saved results as: ", path)

    @classmethod
    def open(cls, filename=None):
        """

        """
        # Select file from file explorer
        if not filename:
            root = Tk()
            root.withdraw()
            root.call('wm', 'attributes', '.', '-topmost', True)
            filename = filedialog.askopenfilename(
                parent=root,
                title='Select results file',
                filetypes=[("Excel Files", ".xlsx")],
                initialdir="../simulations/results")

            if filename == '':
                print('No file chosen, stopping program.')
                sys.exit()

        t = pd.read_excel(filename, sheet_name='time', header=None).to_numpy().transpose()[0]
        displacement = pd.read_excel(filename, sheet_name='displacement', header=None).to_numpy()
        velocity = pd.read_excel(filename, sheet_name='velocity', header=None).to_numpy()
        acceleration = pd.read_excel(filename, sheet_name='acceleration', header=None).to_numpy()

        print("Loading results from: ", filename)

        return cls(t, displacement, velocity, acceleration, filename)
    # endregion

    def options(self):
        # Options for processing the results
        choices = ['Finish viewing results',
                   'Save results',
                   'Print maxRMS at midspan',
                   'Plot maxRMS at midspan',
                   'Cancel']
        question = [inquirer.List('next', message="How would you like to proceed?", choices=choices)]
        answer = inquirer.prompt(question)

        while answer['next'] != 'Finish viewing results':
            if answer['next'] == 'Save results':
                self.save()
            elif answer['next'] == 'Print maxRMS at midspan':
                self.printMaxMidspanRMS()
            elif answer['next'] == 'Plot maxRMS at midspan':
                self.plotMidspanAcceleration()
            elif answer['next'] == 'Cancel':
                sys.exit()
            answer = inquirer.prompt(question)

    # region Process Results
    def printMaxMidspanRMS(self):
        print(f"Max RMS: {self.maxMidspanRMS:.6f} m/s^2\n")

    def plotMidspanAcceleration(self, title='Acceleration'):
        plt.figure(figsize=(9, 4))

        # creating the bar plot
        plt.plot(self.t, self.midspanAcceleration, 'r', self.t, self.midspanRMS, 'b')

        plt.xlabel("Time (s)")
        plt.ylabel("Acceleration (m/s^2)")
        plt.title(title)

        # plt.xlim([0, 40])
        plt.show()

    def calculateRMS(self, x):
        accelerationAtX = self.acceleration[:, x]
        rms = self.timeRMS(accelerationAtX)
        return rms

    def timeRMS(self, x, RMS_Window=1):
        # This function returns the tspan-rms of the signal

        n = len(x)
        i = 0
        while self.t[i] < RMS_Window:
            i += 1
        Npts = i
        rNpts = math.sqrt(Npts)

        rms = np.zeros(n)

        i = 1
        while i < Npts:
            vec = x[0:i]
            rms[i - 1] = np.linalg.norm(vec) / math.sqrt(i)
            i += 1

        while i < n:
            vec = x[i - Npts:i]
            rms[i - 1] = np.linalg.norm(vec) / rNpts
            i += 1

        return rms
    # endregion


if __name__ == '__main__':
    loadResults()

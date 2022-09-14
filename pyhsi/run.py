from crowd import *
from fe_mf import *
import inquirer
import csv
import pprint
import tkinter as tk
from tkinter import filedialog


def main1():
    runParameters = getRunParameters()
    print(runParameters)

    humanProperties = getHumanProperties()
    updateHumanProperties(humanProperties)

    # Crowd parameters
    density = 2
    length = 50
    width = 1
    sync = 0

    # crowd = SinglePedestrian()
    # crowd = DeterministicCrowd(density, length, width, sync)
    # crowd = RandomCrowd(density, length, width, sync)

    m = 73.85
    k = 14.11e3
    xi = 0.3
    crowd = SinglePedestrian(m, xi * 2 * math.sqrt(k * m), k, 2, 0, 0, 1.25, 0)

    fe_mf(crowd)


def runHSI():
    print('running hsi')


def getRunParameters():
    # Bridge Type
    # Bridge properties:
    #   - Number of beam elements
    #   - Length (m)
    #   - Width (m)
    #   - Height (m)
    #   - Young's Modulus (N/m^2)
    #   - Model damping ratio
    #   - Higher mode for damping matrix
    #   - Cross sectional area (m^2)
    #   - Linear Mass (kg/m)
    #   - Beam frequency (Hz)

    # Crowd type
    # Crowd properties:
    #   - Number of people
    #   - Length of crowd
    #   - Width of crowd
    #   - Percent synchronised

    # Human Properties
    # Pedestrian Model

    crowdTypeChoices = ['Single Pedestrian', 'Deterministic Crowd', 'Single Random Crowd', '100 Random Crowds']
    acceptDefaultChoices = ['Yes', 'No']
    pedestrianModelChoices = ['Moving Mass', 'Spring Mass Damper', 'Moving Force']
    modelTypeChoices = ['Modal Analysis', 'Finite Element']

    questions = [
        inquirer.List('crowdType',
                      message="Which crowd type would you like?",
                      choices=crowdTypeChoices,
                      ),
        inquirer.List('acceptDefault',
                      message="Accept default human properties?",
                      choices=acceptDefaultChoices,
                      ),
        inquirer.Checkbox('pedestrianModel',
                          message='Which pedestrian model(s) would you like to use?',
                          choices=pedestrianModelChoices,
                          ),
        inquirer.Checkbox('modelType',
                          message='Which model type(s) would you like to use?',
                          choices=modelTypeChoices,
                          ),
    ]

    runParameters = inquirer.prompt(questions)

    return runParameters


def main():
    sim = SimulationSetup()
    sim.run()
    # sim = Simulation.quickLoad('../simulations/sim1.csv')


class SimulationSetup:
    filename = ''
    beamProperties = {}
    crowdOptions = {}
    humanProperties = {}
    pedestrianModels = []
    modelTypes = []

    def __init__(self, filename=None):
        if not filename:
            self.populate()
        else:
            self.loadSimulation(filename)
        self.next()

    def __str__(self):
        # Display the crowd properties in a readable format
        simRepresentation = '--------------------------------------------------\n'
        simRepresentation += 'Name: {filename}\n'.format(filename=self.filename)
        simRepresentation += '--------------------------------------------------\n'

        simRepresentation += '-Beam-\n'
        for i in self.beamProperties:
            simRepresentation += '{property}: {value}\n'.format(property=i, value=self.beamProperties[i])
        simRepresentation += '--------------------------------------------------\n'

        simRepresentation += '-Crowd Options-\n'
        for i in self.crowdOptions:
            simRepresentation += '{property}: {value}\n'.format(property=i, value=self.crowdOptions[i])
        simRepresentation += '--------------------------------------------------\n'

        simRepresentation += '-Human Properties-\n'
        for i in self.humanProperties:
            simRepresentation += '{property}: {value}\n'.format(property=i, value=self.humanProperties[i])
        simRepresentation += '--------------------------------------------------\n'

        simRepresentation += '-Pedestrian Models-\n'
        for i in self.pedestrianModels:
            simRepresentation += i + '\n'
        simRepresentation += '--------------------------------------------------\n'

        simRepresentation += '-Model Types-\n'
        for i in self.modelTypes:
            simRepresentation += i + '\n'
        simRepresentation += '--------------------------------------------------\n'

        return simRepresentation

    def run(self):
        pass

    def populate(self):
        # Get the simulation properties
        choices = ['Create a new simulation', 'Load simulation']
        question = [inquirer.List('simSource', message="How would you like to start?", choices=choices)]
        answer = inquirer.prompt(question)

        if answer['simSource'] == "Create a new simulation":
            self.createSimulation()
        elif answer['simSource'] == "Load simulation":
            self.loadSimulation()

    def next(self):
        # Options now the simulation properties are loaded
        choices = ['Run the simulation', 'Edit the simulation properties', 'View the simulation properties', 'Cancel']
        question = [inquirer.List('next', message="How would you like to proceed?", choices=choices)]
        answer = inquirer.prompt(question)

        while answer['next'] != 'Run the simulation':
            if answer['next'] == 'Edit the simulation properties':
                self.editSimulation()
            elif answer['next'] == 'View the simulation properties':
                print(self)
            elif answer['next'] == 'Cancel':
                return
            answer = inquirer.prompt(question)

    def createSimulation(self):
        # TODO: Finish method
        self.enterBeamProperties()
        self.enterCrowdOptions()
        self.enterHumanProperties()
        self.enterPedestrianModels()
        self.enterModelTypes()

        # Output properties?

        saveMessage = 'Would you like to save this simulation configuration?'
        saveChoices = ["Save as", "Don't save", "Edit simulation configuration"]
        saveQuestion = [inquirer.List('save', message=saveMessage, choices=saveChoices)]
        saveAnswer = inquirer.prompt(saveQuestion)

        if saveAnswer['save'] == "Save as":
            self.saveSimulation()
        elif saveAnswer['save'] == "Edit simulation configuration":
            self.editSimulation()

    def saveSimulation(self):
        path = ''
        if self.filename != '':
            path = self.filename
            # overwriteMessage = "Would you like to overwrite the file saved at {filename} or save as a new file?"
            # overwriteChoices = ["Overwrite", "Save as new file"]
            # overwriteQuestion = [inquirer.List('overwrite', message=overwriteMessage, choices=overwriteChoices)]
            # overwriteAnswer = inquirer.prompt(overwriteQuestion)
            # if overwriteAnswer == "Overwrite":
            #     path = self.filename

        if path == '':
            filenameMessage = "Enter a filename"
            filenameQuestion = [inquirer.Text('filename', message=filenameMessage)]
            filenameAnswer = inquirer.prompt(filenameQuestion)
            path = '../simulations/{filename}.csv'.format(filename=filenameAnswer['filename'])
            # TODO: Check if file already exists
            self.filename = path

        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)

            # Beam
            writer.writerow(['Beam'])
            for i in self.beamProperties:
                writer.writerow([i, self.beamProperties[i]])
            writer.writerow('')

            # Crowd
            writer.writerow(['Crowd'])
            for i in self.crowdOptions:
                writer.writerow([i, self.crowdOptions[i]])
            writer.writerow('')

            # Human Properties
            writer.writerow(['Human Properties'])
            for i in self.humanProperties:
                writer.writerow([i, self.humanProperties[i]])
            writer.writerow('')

            # Pedestrian Models
            writer.writerow(['Pedestrian Models'])
            for i in self.pedestrianModels:
                writer.writerow([i])
            writer.writerow('')

            # Model Types
            writer.writerow(['Model Types'])
            for i in self.modelTypes:
                writer.writerow([i])

            f.close()

    def loadSimulation(self, filename=None):
        # Get the file to load the simulation from
        # TODO: Allow user to select file from file explorer
        if not filename:
            filenameMessage = "Enter the filename"
            filenameQuestion = [inquirer.Text('filename', message=filenameMessage)]
            filenameAnswer = inquirer.prompt(filenameQuestion)
            filename = '../simulations/{filename}'.format(filename=filenameAnswer['filename'])
            if not filename[-4:] == '.csv':
                filename += '.csv'

        # TODO: Check if the file exists
        print('Loading simulation from: {filename}'.format(filename=filename))
        self.filename = filename

        # Read from the file
        with open(filename, newline='') as csvFile:
            csvReader = csv.reader(csvFile, delimiter=',')

            # beamProperties
            prop = ''
            for row in csvReader:
                if not row:
                    prop = ''
                    continue

                if prop == 'Beam':
                    self.beamProperties[row[0]] = row[1]

                elif prop == 'Crowd':
                    self.crowdOptions[row[0]] = row[1]

                elif prop == 'Human Properties':
                    self.humanProperties[row[0]] = row[1]

                elif prop == 'Pedestrian Models':
                    self.pedestrianModels.append(row[0])

                elif prop == 'Model Types':
                    self.modelTypes.append(row[0])

                props = ['Beam', 'Crowd', 'Human Properties', 'Pedestrian Models', 'Model Types']
                if row[0] in props:
                    prop = row[0]

    def editSimulation(self):
        editMessage = 'What would like to edit?'
        editChoices = ["Stop Editing", "Beam Properties", "Crowd Options", "Human Properties", "Pedestrian Models", "Model Types"]
        editQuestion = [inquirer.List('edit', message=editMessage, choices=editChoices)]
        editAnswer = inquirer.prompt(editQuestion)

        while editAnswer['edit'] != "Stop Editing":
            if editAnswer['edit'] == 'Beam Properties':
                self.enterBeamProperties()
            elif editAnswer['edit'] == 'Crowd Options':
                self.enterCrowdOptions()
            elif editAnswer['edit'] == 'Human Properties':
                self.enterHumanProperties()
            elif editAnswer['edit'] == 'Pedestrian Models':
                self.enterPedestrianModels()
            elif editAnswer['edit'] == 'Model Types':
                self.enterModelTypes()

            editAnswer = inquirer.prompt(editQuestion)

        saveMessage = 'Would you like to save this simulation configuration?'
        saveChoices = ["Save", "Save as", "Don't save"]
        saveQuestion = [inquirer.List('save', message=saveMessage, choices=saveChoices)]
        saveAnswer = inquirer.prompt(saveQuestion)

        if saveAnswer['save'] == "Save":
            self.saveSimulation()
        elif saveAnswer['save'] == "Save as":
            self.filename = ''
            self.saveSimulation()
        elif saveAnswer['save'] == "Edit simulation configuration":
            self.editSimulation()

    # region Enter properties
    def enterBeamProperties(self):
        # TODO: Check wording of questions
        # Ask the user whether they want to import the default beam properties
        loadDefaultBeamProperties = self.loadDefaultQuestion('beam properties')
        if loadDefaultBeamProperties:
            self.loadDefaultBeamProperties()
        else:
            numElementsMessage = 'Number of beam elements'
            lengthMessage = 'Length of the beam (m)'
            widthMessage = 'Width of the beam (m)'
            heightMessage = 'height of the beam (m)'
            EMessage = "Young's modulus of the beam (N/m^2)"
            modalDampingRatioMessage = 'Modal damping ratio of the beam'
            nHighMessage = 'Higher mode for damping matrix'
            areaMessage = 'Cross-section area of the beam (m^2)'
            linearMassMessage = 'Linear mass of the beam (kg/m)'
            beamFreqMessage = 'Beam frequency (Hz)'

            humanPropertiesQuestions = [
                inquirer.Text('meanMass', message=numElementsMessage),
                inquirer.Text('sdMass', message=lengthMessage),
                inquirer.Text('meanPace', message=widthMessage),
                inquirer.Text('sdPace', message=heightMessage),
                inquirer.Text('meanStride', message=EMessage),
                inquirer.Text('sdStride', message=modalDampingRatioMessage),
                inquirer.Text('meanStiffness', message=nHighMessage),
                inquirer.Text('sdStiffness', message=areaMessage),
                inquirer.Text('meanDamping', message=linearMassMessage),
                inquirer.Text('sdDamping', message=beamFreqMessage)
            ]

            humanPropertiesAnswers = inquirer.prompt(humanPropertiesQuestions)

            for i in humanPropertiesAnswers:
                self.humanProperties[i] = humanPropertiesAnswers[i]

    def enterCrowdOptions(self):
        # Get crowd type
        crowdTypeMessage = 'What type of crowd would you like to simulate?'
        crowdTypeChoices = ['Single Pedestrian', 'Deterministic Crowd', 'Random Crowd', 'n Random Crowds']
        crowdTypeQuestion = [inquirer.List('type', message=crowdTypeMessage, choices=crowdTypeChoices)]
        crowdTypeAnswer = inquirer.prompt(crowdTypeQuestion)

        self.crowdOptions['type'] = crowdTypeAnswer['type']
        crowdType = crowdTypeAnswer['type']

        # Get crowd dimensions
        crowdPropertiesQuestions = []

        numPedestriansMessage = 'How many pedestrians are in the crowd?'
        crowdLengthMessage = 'What is the length of the crowd?'
        crowdWidthMessage = 'What is the width of the crowd?'
        percentSynchronisedMessage = 'What percentage of pedestrians in the crowd are synchronised?'

        if crowdType == 'n Random Crowds':
            # Change the wording and add number of crowds
            numCrowdsMessage = 'How many random crowds?'
            numPedestriansMessage = 'How many pedestrians are in each crowd?'
            crowdLengthMessage = 'What is the length of each crowd?'
            crowdWidthMessage = 'What is the width of each crowd?'
            percentSynchronisedMessage = 'What percentage of pedestrians in each crowd are synchronised?'

            crowdPropertiesQuestions.append(inquirer.Text('numCrowds', message=numCrowdsMessage))

        if crowdType != 'Single Pedestrian':
            # Ask the user if they want to import the default deterministic crowd
            loadDefaultCrowdDimensions = self.loadDefaultQuestion('crowd dimensions')
            if loadDefaultCrowdDimensions:
                self.loadDefaultCrowdDimensions()
            else:
                # Ask the questions
                crowdPropertiesQuestions.append(inquirer.Text('numPedestrians', message=numPedestriansMessage))
                crowdPropertiesQuestions.append(inquirer.Text('crowdLength', message=crowdLengthMessage))
                crowdPropertiesQuestions.append(inquirer.Text('crowdWidth', message=crowdWidthMessage))
                crowdPropertiesQuestions.append(inquirer.Text('percentSynchronised', message=percentSynchronisedMessage))

                crowdPropertiesAnswers = inquirer.prompt(crowdPropertiesQuestions)

                for i in crowdPropertiesAnswers:
                    self.crowdOptions[i] = crowdPropertiesAnswers[i]

    def enterHumanProperties(self):
        # TODO: Check wording and units
        # Ask the user whether they want to import default human properties
        loadDefaultHumanProperties = self.loadDefaultQuestion('human properties')
        if loadDefaultHumanProperties:
            self.loadDefaultHumanProperties()
        else:
            meanMassMessage = 'Mean mass of a pedestrian (kg)'
            sdMassMessage = 'Standard deviation of mass of a pedestrian (kg)'
            meanPaceMessage = 'Mean pace of a pedestrian (m/s)'
            sdPaceMessage = 'Standard deviation of pace of a pedestrian (m/s)'
            meanStrideMessage = 'Mean stride of a pedestrian (m)'
            sdStrideMessage = 'Standard deviation of the stride of a pedestrian (m)'
            meanStiffnessMessage = 'Mean stiffness of a pedestrian (Nm?)'
            sdStiffnessMessage = 'Standard deviation of stiffness of a pedestrian (Nm?)'
            meanDampingMessage = 'Mean damping of a pedestrian (Ns/m?)'
            sdDampingMessage = 'Standard deviation of damping of a pedestrian (Ns/m)'

            humanPropertiesQuestions = [
                inquirer.Text('meanMass', message=meanMassMessage),
                inquirer.Text('sdMass', message=sdMassMessage),
                inquirer.Text('meanPace', message=meanPaceMessage),
                inquirer.Text('sdPace', message=sdPaceMessage),
                inquirer.Text('meanStride', message=meanStrideMessage),
                inquirer.Text('sdStride', message=sdStrideMessage),
                inquirer.Text('meanStiffness', message=meanStiffnessMessage),
                inquirer.Text('sdStiffness', message=sdStiffnessMessage),
                inquirer.Text('meanDamping', message=meanDampingMessage),
                inquirer.Text('sdDamping', message=sdDampingMessage)
            ]

            humanPropertiesAnswers = inquirer.prompt(humanPropertiesQuestions)

            for i in humanPropertiesAnswers:
                self.humanProperties[i] = humanPropertiesAnswers[i]

    def enterPedestrianModels(self):
        pedestrianModelMessage = 'Which model type(s) would you like to use?'
        pedestrianModelChoices = ['Moving Mass', 'Moving Force', 'Spring Mass Damper']
        pedestrianModelQuestion = [
            inquirer.Checkbox('pedestrianModel', message=pedestrianModelMessage, choices=pedestrianModelChoices,)
        ]
        pedestrianModelAnswer = inquirer.prompt(pedestrianModelQuestion)

        self.pedestrianModels = pedestrianModelAnswer['pedestrianModel']

    def enterModelTypes(self):
        modelTypesMessage = 'Which pedestrian model(s) would you like to use?'
        modelTypesChoices = ['Modal Analysis', 'Finite Element']
        modelTypesQuestion = [
            inquirer.Checkbox('modelTypes', message=modelTypesMessage, choices=modelTypesChoices)
        ]
        modelTypesAnswer = inquirer.prompt(modelTypesQuestion)

        self.modelTypes = modelTypesAnswer['modelTypes']
    # endregion

    # region Load from default
    @staticmethod
    def loadDefaultQuestion(name):
        loadDefaultMessage = 'Would you like to load the default {name}?'.format(name=name)
        loadDefaultChoices = ['Yes', 'No']
        loadDefaultQuestion = [inquirer.List('loadDefault', message=loadDefaultMessage, choices=loadDefaultChoices)]
        loadDefaultAnswer = inquirer.prompt(loadDefaultQuestion)
        if loadDefaultAnswer['loadDefault'] == 'Yes':
            return True
        elif loadDefaultAnswer['loadDefault'] == 'No':
            return False

    def loadDefaultBeamProperties(self):
        with open('../simulations/defaults/DefaultBeamProperties.csv', newline='') as csvFile:
            csvReader = csv.reader(csvFile, delimiter=',')
            for row in csvReader:
                self.beamProperties[row[0]] = float(row[1])

    def loadDefaultHumanProperties(self):
        with open('../simulations/defaults/DefaultHumanProperties.csv', newline='') as csvFile:
            csvReader = csv.reader(csvFile, delimiter=',')
            for row in csvReader:
                self.humanProperties[row[0]] = float(row[1])

    def loadDefaultCrowdDimensions(self):
        with open('../simulations/defaults/DefaultCrowdDimensions.csv', newline='') as csvFile:
            csvReader = csv.reader(csvFile, delimiter=',')
            for row in csvReader:
                self.crowdOptions[row[0]] = float(row[1])

    # endregion

    @classmethod
    def quickLoad(cls, filename):
        return cls(filename)


if __name__ == '__main__':
    main()

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
    crowd = SinglePedestrian()

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
        # TODO: Do we want to run the simulations from here or pass the object to the solver?
        # Update the human properties of the crowd
        updateHumanProperties(self.humanProperties)

        # Generate the beam
        # TODO: Add beam properties to the beam
        beam = Beam()

        # Generate the crowd
        crowdType = self.crowdOptions['type']
        if crowdType == "Single Pedestrian":
            crowd = SinglePedestrian()
        elif crowdType == "Deterministic Crowd":
            crowd = DeterministicCrowd(
                self.crowdOptions['numPedestrians'],
                self.crowdOptions['crowdLength'],
                self.crowdOptions['crowdWidth'],
                self.crowdOptions['percentSynchronised'])
        elif crowdType == "Random Crowd":
            crowd = RandomCrowd(
                self.crowdOptions['numPedestrians'],
                self.crowdOptions['crowdLength'],
                self.crowdOptions['crowdWidth'],
                self.crowdOptions['percentSynchronised'])
        elif crowdType == "n Random Crowds":
            # TODO: Implement n Random Crowds
            print("Not implemented")

        # Run the simulation for each model combination
        if "Finite Element" in self.modelTypes:
            if "Moving Mass" in self.pedestrianModels:
                # FE MM
                print("Solving system with a 'Finite Element - Moving Mass' model")
                # results = FeMMSolver(crowd, beam)
            if "Moving Force" in self.pedestrianModels:
                # FE MF
                print("Solving system with a 'Finite Element - Moving Force' model")
                # results = FeMfSolver(crowd, beam)
            if "Spring Mass Damper" in self.pedestrianModels:
                # FE SMD
                print("Solving system with a 'Finite Element - Spring Mass Damper' model")
                # results = FeSMDSolver(crowd, beam)
        if "Modal Analysis" in self.modelTypes:
            if "Moving Mass" in self.pedestrianModels:
                # MO MM
                print("Solving system with a 'Modal Analysis - Moving Mass' model")
                # results = MoMMSolver(crowd, beam)
            if "Moving Force" in self.pedestrianModels:
                # MO MF
                print("Solving system with a 'Modal Analysis - Moving Force' model")
                # results = MoMfSolver(crowd, beam)
            if "Spring Mass Damper" in self.pedestrianModels:
                # MO SMD
                print("Solving system with a 'Modal Analysis - Sprint Mass Damper' model")
                # results = MoSMDSolver(crowd, beam)

        # TODO: Determine how to present results

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

            # Add the current value if editing
            numElementsDefault = self.beamProperties['numElements'] if 'numElements' in self.beamProperties else None
            lengthDefault = self.beamProperties['length'] if 'length' in self.beamProperties else None
            widthDefault = self.beamProperties['width'] if 'width' in self.beamProperties else None
            heightDefault = self.beamProperties['height'] if 'height' in self.beamProperties else None
            EDefault = self.beamProperties['E'] if 'E' in self.beamProperties else None
            modalDampingRatioDefault = self.beamProperties['modalDampingRatio'] if 'modalDampingRatio' in self.beamProperties else None
            nHighDefault = self.beamProperties['nHigh'] if 'nHigh' in self.beamProperties else None
            areaDefault = self.beamProperties['area'] if 'area' in self.beamProperties else None
            linearDefault = self.beamProperties['linearMass'] if 'linearMass' in self.beamProperties else None
            beamFreqDefault = self.beamProperties['beamFreq'] if 'beamFreq' in self.beamProperties else None

            beamPropertiesQuestions = [
                inquirer.Text('numElements', message=numElementsMessage, default=numElementsDefault),
                inquirer.Text('length', message=lengthMessage, default=lengthDefault),
                inquirer.Text('width', message=widthMessage, default=widthDefault),
                inquirer.Text('height', message=heightMessage, default=heightDefault),
                inquirer.Text('E', message=EMessage, default=EDefault),
                inquirer.Text('modalDampingRatio', message=modalDampingRatioMessage, default=modalDampingRatioDefault),
                inquirer.Text('nHigh', message=nHighMessage, default=nHighDefault),
                inquirer.Text('area', message=areaMessage, default=areaDefault),
                inquirer.Text('linearMass', message=linearMassMessage, default=linearDefault),
                inquirer.Text('beamFreq', message=beamFreqMessage, default=beamFreqDefault)
            ]

            beamPropertiesAnswers = inquirer.prompt(beamPropertiesQuestions)

            for i in beamPropertiesAnswers:
                self.humanProperties[i] = beamPropertiesAnswers[i]

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

        # Add the current value if editing
        numCrowdsDefault = self.crowdOptions['numCrowds'] if 'numCrowds' in self.crowdOptions else None
        numPedestriansDefault = self.crowdOptions['numPedestrians'] if 'numPedestrians' in self.crowdOptions else None
        crowdLengthDefault = self.crowdOptions['crowdLength'] if 'crowdLength' in self.crowdOptions else None
        crowdWidthDefault = self.crowdOptions['crowdWidth'] if 'crowdWidth' in self.crowdOptions else None
        percentSynchronisedDefault = self.crowdOptions['percentSynchronised'] if 'percentSynchronised' in self.crowdOptions else None

        if crowdType == 'n Random Crowds':
            # Change the wording and add number of crowds
            numCrowdsMessage = 'How many random crowds?'
            numPedestriansMessage = 'How many pedestrians are in each crowd?'
            crowdLengthMessage = 'What is the length of each crowd?'
            crowdWidthMessage = 'What is the width of each crowd?'
            percentSynchronisedMessage = 'What percentage of pedestrians in each crowd are synchronised?'

            crowdPropertiesQuestions.append(inquirer.Text('numCrowds', message=numCrowdsMessage, default=numCrowdsDefault))

        if crowdType != 'Single Pedestrian':
            # Ask the user if they want to import the default deterministic crowd
            loadDefaultCrowdDimensions = self.loadDefaultQuestion('crowd dimensions')
            if loadDefaultCrowdDimensions:
                self.loadDefaultCrowdDimensions()
            else:
                # Ask the questions
                crowdPropertiesQuestions.append(inquirer.Text('numPedestrians', message=numPedestriansMessage, default=numPedestriansDefault))
                crowdPropertiesQuestions.append(inquirer.Text('crowdLength', message=crowdLengthMessage, default=crowdLengthDefault))
                crowdPropertiesQuestions.append(inquirer.Text('crowdWidth', message=crowdWidthMessage, default=crowdWidthDefault))
                crowdPropertiesQuestions.append(inquirer.Text('percentSynchronised', message=percentSynchronisedMessage, default=percentSynchronisedDefault))

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

            meanMassDefault = self.humanProperties['meanMass'] if 'meanMass' in self.humanProperties else None
            sdMassDefault = self.humanProperties['sdMass'] if 'sdMass' in self.humanProperties else None
            meanPaceDefault = self.humanProperties['meanPace'] if 'meanPace' in self.humanProperties else None
            sdPaceDefault = self.humanProperties['sdPace'] if 'sdPace' in self.humanProperties else None
            meanStrideDefault = self.humanProperties['meanStride'] if 'meanStride' in self.humanProperties else None
            sdStrideDefault = self.humanProperties['sdStride'] if 'sdStride' in self.humanProperties else None
            meanStiffnessDefault = self.humanProperties['meanStiffness'] if 'meanStiffness' in self.humanProperties else None
            sdStiffnessDefault = self.humanProperties['sdStiffness'] if 'sdStiffness' in self.humanProperties else None
            meanDampingDefault = self.humanProperties['meanDamping'] if 'meanDamping' in self.humanProperties else None
            sdDampingDefault = self.humanProperties['sdDamping'] if 'sdDamping' in self.humanProperties else None

            humanPropertiesQuestions = [
                inquirer.Text('meanMass', message=meanMassMessage, default=meanMassDefault),
                inquirer.Text('sdMass', message=sdMassMessage, default=sdMassDefault),
                inquirer.Text('meanPace', message=meanPaceMessage, default=meanPaceDefault),
                inquirer.Text('sdPace', message=sdPaceMessage, default=sdPaceDefault),
                inquirer.Text('meanStride', message=meanStrideMessage, default=meanStrideDefault),
                inquirer.Text('sdStride', message=sdStrideMessage, default=sdStrideDefault),
                inquirer.Text('meanStiffness', message=meanStiffnessMessage, default=meanStiffnessDefault),
                inquirer.Text('sdStiffness', message=sdStiffnessMessage, default=sdStiffnessDefault),
                inquirer.Text('meanDamping', message=meanDampingMessage, default=meanDampingDefault),
                inquirer.Text('sdDamping', message=sdDampingMessage, default=sdDampingDefault)
            ]

            humanPropertiesAnswers = inquirer.prompt(humanPropertiesQuestions)

            for i in humanPropertiesAnswers:
                self.humanProperties[i] = humanPropertiesAnswers[i]

    def enterPedestrianModels(self):
        pedestrianModelMessage = 'Which model type(s) would you like to use?'
        pedestrianModelChoices = ['Moving Mass', 'Moving Force', 'Spring Mass Damper']
        pedestrianModelDefaults = self.pedestrianModels
        pedestrianModelQuestion = [
            inquirer.Checkbox('pedestrianModel', message=pedestrianModelMessage, choices=pedestrianModelChoices, default=pedestrianModelDefaults)
        ]
        pedestrianModelAnswer = inquirer.prompt(pedestrianModelQuestion)

        self.pedestrianModels = pedestrianModelAnswer['pedestrianModel']

    def enterModelTypes(self):
        modelTypesMessage = 'Which pedestrian model(s) would you like to use?'
        modelTypesChoices = ['Finite Element', 'Modal Analysis']
        modelTypesDefaults = self.modelTypes
        modelTypesQuestion = [
            inquirer.Checkbox('modelTypes', message=modelTypesMessage, choices=modelTypesChoices, default=modelTypesDefaults)
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

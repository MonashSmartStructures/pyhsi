import sys

from crowd import *
from solver import *
from results import *
import inquirer
import csv

from tkinter import *
from tkinter import filedialog


def main():
    # Run a simulation or load results
    # # Get the simulation properties
    # choices = ['Run a simulation', 'Load results']
    # question = [inquirer.List('simSource', message="How would you like to start?", choices=choices)]
    # answer = inquirer.prompt(question)
    #
    # if answer['simSource'] == "Create a new simulation":
    #     self.createSimulation()
    # elif answer['simSource'] == "Load simulation":
    #     self.loadSimulation()

    # Setup Simulation
    sim = SimulationSetup()
    # sim = SimulationSetup('../simulations/setups/crowd1.csv')  # Quick run
    solvers = sim.loadSolvers()

    # Run Simulations
    results = {}
    for i in solvers:
        solvers[i].solve()
        t, q, dq, ddq = solvers[i].getResults()
        pedModel, modelType = solvers[i].getModelType()
        results[i] = Results(t, q, dq, ddq, pedModel, modelType, sim.filename)
        results[i].askSave()
        results[i].options()

    # Process Results
    # results['FE_MM'].calcMaxRms()

    # sim.run()


def testRun():
    Solver.setNumSteps(1)

    # sim = SimulationSetup()
    sim = SimulationSetup('../simulations/setups/test2.csv')
    solvers = sim.loadSolvers()
    solvers['FE_MM'].solve()
    t, q, dq, ddq = solvers['FE_MM'].getResults()
    pedModel, modelType = solvers['FE_MM'].getModelType()
    results = Results(t, q, dq, ddq, pedModel, modelType, sim.filename)
    results.askSave()
    results.options()


class SimulationSetup:
    filename = ''
    beamProperties = {}
    crowdOptions = {}
    populationProperties = {}
    pedestrianModels = []
    modelTypes = []

    def __init__(self, filename=None):
        if not filename:
            self.populate()
            self.next()
        else:
            self.loadSimulation(filename)

    def createSimulation(self):
        self.enterBeamProperties()
        self.enterCrowdOptions()
        self.enterPopulationProperties()
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

    def loadSolvers(self):
        self.fixAllDataTypes()

        # Update the population properties for crowd generation
        updatePopulationProperties(self.populationProperties)

        # Generate the beam
        # TODO: Add beam properties to the beam
        beam = Beam.fromDict(self.beamProperties)

        # Generate the crowd
        crowd = self.getCrowdClass().fromDict(self.crowdOptions)

        solvers = {}
        # Run the simulation for each model combination
        if "Finite Element" in self.modelTypes:
            if "Moving Mass" in self.pedestrianModels:
                # FE MM
                solvers['FE_MM'] = FeMmSolver(crowd, beam)
            if "Moving Force" in self.pedestrianModels:
                # FE MF
                solvers['FE_MF'] = FeMfSolver(crowd, beam)
            if "Spring Mass Damper" in self.pedestrianModels:
                # FE SMD
                solvers['FE_SMD'] = FeSMDSolver(crowd, beam)
        if "Modal Analysis" in self.modelTypes:
            if "Moving Mass" in self.pedestrianModels:
                # MO MM
                solvers['MO_MM'] = MoMmSolver(crowd, beam)
            if "Moving Force" in self.pedestrianModels:
                # MO MF
                solvers['MO_MF'] = MoMfSolver(crowd, beam)
            if "Spring Mass Damper" in self.pedestrianModels:
                # MO SMD
                solvers['MO_SMD'] = MoSMDSolver(crowd, beam)

        # TODO: Determine how to present results
        return solvers

    def getCrowdClass(self):
        # Generate the crowd
        crowdType = self.crowdOptions['type']
        if crowdType == "Single Pedestrian":
            return SinglePedestrian
        elif crowdType == "Deterministic Crowd":
            return DeterministicCrowd
        elif crowdType == "Random Crowd":
            return RandomCrowd
        elif crowdType == "Exact Crowd":
            return ExactCrowd
        elif crowdType == "n Random Crowds":
            # TODO: Implement n Random Crowds
            print("Not implemented")
        else:
            sys.exit("Invalid crowd")

    # region Navigate class
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
                print(repr(self))
            elif answer['next'] == 'Cancel':
                sys.exit()
            answer = inquirer.prompt(question)
    # endregion

    # region Load Save and Edit
    def saveSimulation(self):
        path = ''
        if self.filename != '':
            path = self.filename

        if path == '':
            filenameMessage = "Enter a filename"
            filenameQuestion = [inquirer.Text('filename', message=filenameMessage)]
            filenameAnswer = inquirer.prompt(filenameQuestion)
            path = f"../simulations/{filenameAnswer['filename']}"
            if not path[-4:] == '.csv':
                path += '.csv'

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

            # Population Properties
            writer.writerow(['Population Properties'])
            for i in self.populationProperties:
                writer.writerow([i, self.populationProperties[i]])
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
        # TODO: Check this works with other OS
        if not filename:
            root = Tk()
            root.withdraw()
            root.call('wm', 'attributes', '.', '-topmost', True)
            filename = filedialog.askopenfilename(
                parent=root,
                title='Select simulation file',
                filetypes=[("Text Files", ".csv")],
                initialdir="../simulations/setups")

            if filename == '':
                print('No file chosen, stopping program.')
                sys.exit()

        print(f"Loading simulation from: {filename}")
        self.filename = filename

        # Read from the file
        with open(filename, newline='') as csvFile:
            csvReader = csv.reader(csvFile, delimiter=',')

            # beamProperties
            prop = ''
            for row in csvReader:
                if not row or row == ['', '']:
                    prop = ''
                    continue

                if prop == 'Beam':
                    self.beamProperties[row[0]] = row[1]

                elif prop == 'Crowd':
                    self.crowdOptions[row[0]] = row[1]

                elif prop == 'Population Properties':
                    self.populationProperties[row[0]] = row[1]

                elif prop == 'Pedestrian Models':
                    self.pedestrianModels.append(row[0])

                elif prop == 'Model Types':
                    self.modelTypes.append(row[0])

                props = ['Beam', 'Crowd', 'Population Properties', 'Pedestrian Models', 'Model Types']
                if row[0] in props:
                    prop = row[0]

    def editSimulation(self):
        editMessage = 'What would like to edit?'
        editChoices = ["Stop Editing", "Beam Properties", "Crowd Options", "Population Properties", "Pedestrian Models", "Model Types"]
        editQuestion = [inquirer.List('edit', message=editMessage, choices=editChoices)]
        editAnswer = inquirer.prompt(editQuestion)

        while editAnswer['edit'] != "Stop Editing":
            if editAnswer['edit'] == 'Beam Properties':
                self.enterBeamProperties()
            elif editAnswer['edit'] == 'Crowd Options':
                self.enterCrowdOptions()
            elif editAnswer['edit'] == 'Population Properties':
                self.enterPopulationProperties()
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
    # endregion

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
                self.beamProperties[i] = beamPropertiesAnswers[i]

        self.fixBeamPropertiesDataTypes()

    def enterCrowdOptions(self):
        # Get crowd type
        crowdTypeMessage = 'What type of crowd would you like to simulate?'
        crowdTypeChoices = ['Single Pedestrian', 'Deterministic Crowd', 'Random Crowd', 'Exact Crowd', 'n Random Crowds']
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

        self.fixCrowdOptionsDataTypes()

    def enterPopulationProperties(self):
        # TODO: Check wording and units
        # Ask the user whether they want to import default population properties
        loadDefaultPopulationProperties = self.loadDefaultQuestion('population properties')
        if loadDefaultPopulationProperties:
            self.loadDefaultPopulationProperties()
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

            meanMassDefault = self.populationProperties['meanMass'] if 'meanMass' in self.populationProperties else None
            sdMassDefault = self.populationProperties['sdMass'] if 'sdMass' in self.populationProperties else None
            meanPaceDefault = self.populationProperties['meanPace'] if 'meanPace' in self.populationProperties else None
            sdPaceDefault = self.populationProperties['sdPace'] if 'sdPace' in self.populationProperties else None
            meanStrideDefault = self.populationProperties['meanStride'] if 'meanStride' in self.populationProperties else None
            sdStrideDefault = self.populationProperties['sdStride'] if 'sdStride' in self.populationProperties else None
            meanStiffnessDefault = self.populationProperties['meanStiffness'] if 'meanStiffness' in self.populationProperties else None
            sdStiffnessDefault = self.populationProperties['sdStiffness'] if 'sdStiffness' in self.populationProperties else None
            meanDampingDefault = self.populationProperties['meanDamping'] if 'meanDamping' in self.populationProperties else None
            sdDampingDefault = self.populationProperties['sdDamping'] if 'sdDamping' in self.populationProperties else None

            populationPropertiesQuestions = [
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

            populationPropertiesAnswers = inquirer.prompt(populationPropertiesQuestions)

            for i in populationPropertiesAnswers:
                self.populationProperties[i] = populationPropertiesAnswers[i]

        self.fixPopulationPropertiesDataTypes()

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
        loadDefaultMessage = f"Would you like to load the default {name}?"
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

    def loadDefaultPopulationProperties(self):
        with open('../simulations/defaults/DefaultPopulationProperties.csv', newline='') as csvFile:
            csvReader = csv.reader(csvFile, delimiter=',')
            for row in csvReader:
                self.populationProperties[row[0]] = float(row[1])

    def loadDefaultCrowdDimensions(self):
        with open('../simulations/defaults/DefaultCrowdDimensions.csv', newline='') as csvFile:
            csvReader = csv.reader(csvFile, delimiter=',')
            for row in csvReader:
                self.crowdOptions[row[0]] = float(row[1])

    # endregion

    # region Fix Data Types
    def fixAllDataTypes(self):
        self.fixBeamPropertiesDataTypes()
        self.fixCrowdOptionsDataTypes()
        self.fixPopulationPropertiesDataTypes()

    def fixBeamPropertiesDataTypes(self):
        for i in self.beamProperties:
            self.beamProperties[i] = float(self.beamProperties[i])
            if i == 'numElements' or i == 'nHigh':
                self.beamProperties[i] = int(self.beamProperties[i])

    def fixCrowdOptionsDataTypes(self):
        for i in self.crowdOptions:
            if i != 'type':
                self.crowdOptions[i] = float(self.crowdOptions[i])
                if i == 'numPedestrians':
                    self.crowdOptions[i] = int(self.crowdOptions[i])

    def fixPopulationPropertiesDataTypes(self):
        for i in self.populationProperties:
            self.populationProperties[i] = float(self.populationProperties[i])
    # endregion

    # region Other Dunder Methods
    def __repr__(self):
        return f"SimulationSetup('{self.filename}')"

    def __str__(self):
        # Display the crowd properties in a readable format
        simRepresentation = '--------------------------------------------------\n'
        simRepresentation += f"Name: {self.filename[15:]}\n"
        simRepresentation += '--------------------------------------------------\n'

        simRepresentation += '-Beam-\n'
        for i in self.beamProperties:
            simRepresentation += f"{i}: {self.beamProperties[i]}\n"
        simRepresentation += '--------------------------------------------------\n'

        simRepresentation += '-Crowd Options-\n'
        for i in self.crowdOptions:
            simRepresentation += f"{i}: {self.crowdOptions[i]}\n"
        simRepresentation += '--------------------------------------------------\n'

        simRepresentation += '-Population Properties-\n'
        for i in self.populationProperties:
            simRepresentation += f"{i}: {self.populationProperties[i]}\n"
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
    # endregion


if __name__ == '__main__':
    main()
    # testRun()

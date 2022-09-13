from crowd import *
from fe_mf import *
import inquirer
import csv
import pprint


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
    sim = Simulation()


class Simulation:
    filename = ''
    beamProperties = {}
    crowdOptions = {}
    humanProperties = {}
    pedestrianModels = []
    modelTypes = []

    def __init__(self):
        # Get the simulation properties
        choices = ['Create a new simulation', 'Load simulation']
        question = [inquirer.List('simSource', message="How would you like to start?", choices=choices)]
        answer = inquirer.prompt(question)

        if answer['simSource'] == "Create a new simulation":
            self.createSimulation()
        elif answer['simSource'] == "Load simulation":
            self.loadSimulation()

        # Options now the simulation properties are loaded
        choices = ['Run the simulation', 'Edit the simulation properties', 'View the simulation properties']
        question = [inquirer.List('next', message="How would you like to proceed?", choices=choices)]
        answer = inquirer.prompt(question)

        while answer['next'] != 'Run the simulation':
            if answer['next'] == 'Edit the simulation properties':
                self.editSimulation()
            elif answer['next'] == 'View the simulation properties':
                print(self)
            answer = inquirer.prompt(question)

    def __str__(self):
        # Display the crowd properties in a readable format
        return "Here are the properties of this simulation."

    def createSimulation(self):
        # TODO: Finish method
        self.enterBeamProperties()
        self.enterCrowdOptions()
        self.enterHumanProperties()
        self.enterPedestrianModels()
        self.enterModelTypes()

        # Output properties?

        saveMessage = 'Whould you like to save this simulation configuration?'
        saveChoices = ["Save as", "Don't save", "Edit simulation configuration"]
        saveQuestion = [inquirer.List('save', message=saveMessage, choices=saveChoices)]
        saveAnswer = inquirer.prompt(saveQuestion)

        if saveAnswer['save'] == "Save as":
            self.saveSimulation()
        elif saveAnswer['save'] == "Edit simulation configuration":
            self.editSimulation()

    def saveSimulation(self):
        # TODO: Write method

        if self.filename != '':
            overwriteMessage = "Would you like to overwrite the file saved at {filename} or save as a new file?"
            overwriteChoices = ["Overwrite", "Save as new file"]
            overwriteQuestion = [inquirer.List('overwrite', message=overwriteMessage, choices=overwriteChoices)]
            overwriteAnswer = inquirer.prompt(overwriteQuestion)

        # Would you like to overwrite
        pass

    def loadSimulation(self):
        # TODO: Write method
        pass

    def editSimulation(self):
        # TODO: Write method
        pass

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
        pedestrianModelChoices = ['Moving Mass', 'Spring Mass Damper', 'Moving Force']
        pedestrianModelQuestion = [
            inquirer.Checkbox('pedestrianModel', message=pedestrianModelMessage, choices=pedestrianModelChoices,)
        ]
        pedestrianModelAnswer = inquirer.prompt(pedestrianModelQuestion)

        self.pedestrianModels = pedestrianModelAnswer['pedestrianModel']

    def enterModelTypes(self):
        modelTypesMessage = 'Which pedestrian model(s) would you like to use?'
        modelTypesChoices = ['Modal Analysis', 'Finite Element']
        modelTypesQuestion = [
            inquirer.Checkbox('modelTypes', message=modelTypesMessage, choices=modelTypesChoices, )
        ]
        modelTypesAnswer = inquirer.prompt(modelTypesQuestion)

        self.pedestrianModels = modelTypesAnswer['modelTypes']
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

    # region Display Attributes
    def displayHumanProperties(self):
        pprint.pprint(self.humanProperties, sort_dicts=False)
    # endregion


if __name__ == '__main__':
    main()

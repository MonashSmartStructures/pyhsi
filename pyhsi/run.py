from crowd import *
import inquirer


def main():

    # runParameters = getRunParameters()
    # print(runParameters)

    humanProperties = getHumanProperties()
    updateHumanProperties(humanProperties)

    # Crowd parameters
    density = 2
    length = 50
    width = 1
    sync = 0

    # crowd = SinglePedestrian()
    # crowd = DeterministicCrowd(density, length, width, sync)
    crowd = RandomCrowd(density, length, width, sync)
    print(crowd)


def runHSI():
    print('running hsi')


def getRunParameters():
    crowdTypeOptions = ['Single Pedestrian', 'Deterministic Crowd', 'Single Random Crowd', '100 Random Crowds']
    acceptDefaultOptions = ['Yes', 'No']
    pedestrianModelOptions = ['Moving Mass', 'Spring Mass Damper', 'Moving Force']
    modelTypeOptions = ['Modal Analysis', 'Finite Element']

    questions = [
        inquirer.List('crowdType',
                      message="Which crowd type would you like?",
                      choices=crowdTypeOptions,
                      ),
        inquirer.List('acceptDefault',
                      message="Accept default human properties?",
                      choices=acceptDefaultOptions,
                      ),
        inquirer.Checkbox('pedestrianModel',
                          message='Which pedestrian model(s) would you like to use?',
                          choices=pedestrianModelOptions,
                          ),
        inquirer.Checkbox('modelType',
                          message='Which model type(s) would you like to use?',
                          choices=modelTypeOptions,
                          ),
    ]

    runParameters = inquirer.prompt(questions)

    return runParameters


if __name__ == '__main__':
    main()

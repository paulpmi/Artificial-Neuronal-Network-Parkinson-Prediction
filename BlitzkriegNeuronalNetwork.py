import numpy
numpy.set_printoptions(suppress=True)
numpy.set_printoptions(threshold=numpy.inf)
numpy.random.seed(1)

neuronsInput = 22
neuronsOutput = 5875
neuronsHidden = 2
nrHidden = 1

def activate(x, synapses):
    s = numpy.dot(x, synapses)
    return 1/(1+numpy.exp(-s))

def derivate(x):
    return x * (1 - x)

def createSynapses():
    synapses = []
    for i in range(neuronsHidden):
        synapses.append(2*numpy.random.random((neuronsInput, neuronsOutput)) -1)
    return synapses

def computeOutputs(trainData):
    outputs = []
    noOutputs = 0
    for t in trainData:
        # if t[-1] not in outputs:
        outputs.append(t[-1])
        noOutputs += 1
    print("outputs ", outputs)
    return outputs

def readData(fileName):
    f = open(fileName, 'r')
    data = []
    out = []
    i = 0
    for line in f:
        data.append(line.strip().split(','))
        for j in range(0, len(data[i]) - 1):
            data[i][j] = float(data[i][j])
            out.append(float(data[i][j]))
        i += 1
    return data

def normaliseData(trainData):
    for j in range(len(trainData[0][:-1])):
        summ = 0.0
        for i in range(len(trainData)):
            summ += trainData[i][j]
        mean = summ / len(trainData)
        squareSum = 0.0
        for i in range(len(trainData)):
            squareSum += (trainData[i][j] - mean) ** 2
        deviation = numpy.sqrt(squareSum / len(trainData))
        for i in range(len(trainData)):
            trainData[i][j] = (trainData[i][j] - mean) / deviation
    return trainData

def checkGlobalErr(err):
    sumerror = 0
    for j in err:
        for i in j:
            sumerror += sum(i)
            accuracy = 1 - (sumerror / len(err))
            if accuracy < 1:
                return True
            else:
                return False

def writeData(filename, str):
    f = open(filename, 'w')
    f.write(str)

input = readData("input.in")
input = normaliseData(input)
input = numpy.asarray(input, dtype=numpy.float64)

trainingOutput = computeOutputs(input)
trainingOutput = numpy.asarray(trainingOutput, dtype=numpy.float64)
stop = False
i = 0
synapses = createSynapses()

while stop == False:
    print(i)
    output = []
    error = []
    adjustments = []
    for j in range(neuronsHidden):
        output.append(activate(input, synapses[j]))
    for j in range(neuronsHidden):
        error.append(trainingOutput - output[j])
    for j in range(len(error)):
        adjustments.append(numpy.dot(input.T, error[j]*derivate(output[j])))
    for j in range(len(error)):
        synapses[j] += adjustments[j]
    i+=1
    stop = checkGlobalErr(error)

# TO DO: LESS INPUT SYNAPSES AND OUTPUT
# TO DO: OOP
writeData("Learning.txt", str(activate([1,40,0,19.701,28.695,35.389,0.00481,2.462e-005,0.00205,0.00208,0.00616,0.01675,0.181,0.00734,0.00844,0.01458,0.02202,0.02022,23.047,0.46222,0.54405,0.21014], synapses)))
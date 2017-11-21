import numpy
numpy.set_printoptions(suppress=True)
numpy.set_printoptions(threshold=numpy.inf)
numpy.random.seed(1)

#0.15336
#0.53984
#0.17793
neuronsInput = 21
neuronsOutput = 1
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
        #print("One line")
        q = []
        for j in range(0, len(data[i]) - 1):
            #print(data[i][j])
            q.append(float(data[i][j]))
            data[i][j] = float(data[i][j])
        out.append(q)
        i += 1
    return out

def normaliseData(trainData):
    for j in range(len(trainData[0])):
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

"""
def checkGlobalErr(err):
    sumerror = 0
    for j in err:
        for i in j:
            sumerror += sum(i)
            accuracy = 1 - (sumerror / len(err))
            print(accuracy)
            if accuracy < 1:
                return True
            else:
                return False

"""
oldacc = 0

def checkGlobalErr(err):
    global oldacc
    sumerror = 0
    for j in err:
        for i in j:
            sumerror += sum(i)
        accuracy = 1 - (sumerror / len(err))
    #print(accuracy)
    #print(oldacc)
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

while stop == False and i < 10000:
    output = []
    error = []
    adjustments = []
    for j in range(neuronsHidden):
        output.append(activate(input, synapses[j]))
    for j in range(neuronsHidden):
        error.append(trainingOutput[j] - output[j])
    for j in range(len(error)):
        adjustments.append(numpy.dot(input.T, error[j]*derivate(output[j])))
    for j in range(len(error)):
        synapses[j] += adjustments[j]
    i+=1
    stop = checkGlobalErr(error)

writeData("Synapses.txt", str(synapses))
# TO DO: LESS INPUT SYNAPSES AND OUTPUT
# TO DO: OOP

writeData("Learning.txt", str(activate([

23,59,1,58.396,13.788,24.767,0.00416,2.286e-005,0.00221,0.00233,0.00664,0.02131,0.187,0.01146,0.01235,0.01708,0.03437,0.012378,22.712,0.42162,0.70732], synapses)))

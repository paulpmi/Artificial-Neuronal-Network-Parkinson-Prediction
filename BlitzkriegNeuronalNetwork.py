import numpy
numpy.set_printoptions(suppress=True)
numpy.set_printoptions(threshold=numpy.inf)
numpy.random.seed(1)

#0.15336
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
    if oldacc == accuracy:
        print(accuracy)
    if accuracy < 1:
        return True
    else:
        oldacc = accuracy
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

while stop == False and i < 3000:
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

# TO DO: LESS INPUT SYNAPSES AND OUTPUT
# TO DO: OOP
writeData("Learning.txt", str(activate([42,61,0,170.73,20.513,31.513,0.00282,2.11e-005,0.00135,0.00166,0.00406,0.01907,0.171,0.00946,0.01154,0.0147,0.02839,0.008172,23.259,0.58608,0.57077], synapses)))
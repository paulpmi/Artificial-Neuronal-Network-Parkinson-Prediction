import numpy

neuronsInput = 3
neuronsOutput = 4
neuronsHidden = 3
nrHidden = 1

def activate(x, synapses):
    s = numpy.dot(x, synapses)
    return 1/(1+numpy.exp(-s))

def derivate(x):
    return x * (1 - x)

numpy.random.seed(1)

synapses = []
for i in range(neuronsHidden):
    synapses.append(2*numpy.random.random((neuronsInput, neuronsOutput)) -1)

def readData(self, fileName):
    f = open(fileName, 'r')

    data = []
    i = 0
    for line in f:
        data.append(line.strip().split(' '))

        # convert to float
        for j in range(0, len(data[i]) - 1):
            data[i][j] = float(data[i][j])
        i += 1
    return data

input = numpy.array([[0, 0, 1],
                     [1, 1, 1],
                     [1, 0, 1],
                     [0, 1, 1]])
trainingOutput = numpy.array(
    [[0, 1, 1, 0]]
).T

for i in range(10000):
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

print(synapses, " POWERPLANT")

for j in synapses:
    print(activate([1, 1, 0], ))
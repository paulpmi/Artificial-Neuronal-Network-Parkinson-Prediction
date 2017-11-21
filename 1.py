import numpy as np

class Neuron:
    def __init__(self, edges):
        self.edges = edges
        np.random.seed(1)
        self.weights = np.array([2*np.random.random() - 1 for _ in range(edges)])
        self.output = 0
        self.error = 0

    def activateSigmoid(self, input):
        x = np.dot(input, self.weights)
        return 1/(1+np.exp(-x))

    def activateTahn(self, input):
        x = np.dot(input, self.weights)
        return 1 - np.tanh(x)**2

class Layer:
    def __init__(self, neurons, inputs):
        self.nrNeurons = neurons
        self.inputs = inputs
        self.neurons = np.array([Neuron(inputs) for _ in range(neurons)])
        self.weights = np.array([self.neurons[i].weigths for i in range(neurons)])

    def derivate(self, x):
        return x*(1 - x)

class Network:
    def __init__(self, inputNeurons, outputNeurons, hiddenLayers, nrNeuronsOnHiddenLayer):
        self.inputNeurons = inputNeurons
        self.outputNeurons = outputNeurons
        self.hiddenLayers = hiddenLayers
        self.nrOfHiddenNeurons = nrNeuronsOnHiddenLayer
        self.Layers = []
        self.propagation = []

        self.LEARNING_RATE = 0.01
        self.MOMENTUM = 0.9

        self.createNetwork()

    def createNetwork(self):
        self.Layers = np.array([Layer(self.inputNeurons, 0)])
        self.Layers = np.array([Layer(self.nrOfHiddenNeurons, self.inputNeurons)])
        self.Layers += np.array([Layer(self.nrOfHiddenNeurons, self.nrOfHiddenNeurons) for _ in range(self.hiddenLayers-1)])
        self.Layers += np.array([Layer(self.outputNeurons, self.nrOfHiddenNeurons)])

        self.propagation = [0 for _ in range(len(self.Layers))]


    def forwardPropagate(self):
        for i in range(1, len(self.Layers)):
            layerWeights = 0
            for j in range(len(self.Layers.neurons)):
                layerWeights += self.Layers[i].neurons[j].weights
            self.propagation[i-1] = self.activateSigmoid(np.dot(self.Layers[i-1], self.Layers[i].weights))

    def activateSigmoid(self, input):
        x = np.dot(input, self.weights)
        return 1/(1+np.exp(-x))

    def activateTahn(self, input):
        x = np.dot(input, self.weights)
        return 1 - np.tanh(x)**2

    def error(self):
        for i in range(len(self.Layers)-1, 0):
            pass


import numpy

print(numpy.exp(-numpy.dot(numpy.random.random((4,3)), numpy.random.random((3,4)))))
import numpy

class Neuron:
    def __init__(self, incoming, outgoing):
        self.incoming = incoming
        self.outgoing = outgoing
        self.synapticWeights = 2*numpy.random.random((incoming, outgoing)) - 1
        #print(self.synapticWeights)
        self.output = 0

    def sigmoid(self, information):
        net = numpy.dot(information, self.synapticWeights)
        self.output = 1/(1+numpy.exp(-net))
        return self.output

    def __str__(self):
        return self.output

class Network:
    def __init__(self, inputNeurons, outputNeurons, hiddenNeutorn, nrHiddenLayers, trainingInputData):
        # initialize data
        numpy.random.seed(1)
        self.inputNeurons, self.outputNeurons, self.hiddenNeutorn, self.nrHiddenLayers = inputNeurons, outputNeurons, hiddenNeutorn, nrHiddenLayers
        self.trainingInputData = trainingInputData
        self.Layers = []

        self.createLayers()

    def createLayers(self):
        # create Layers...not sure on first line if it is correct
        #LayerInput = 2*numpy.random.random((1, self.hiddenNeutorn)) - 1
        #LayerInputHidden = [2*numpy.random.random((self.inputNeurons, self.hiddenNeutorn)) - 1 for i in range(self.hiddenNeutorn)]
        #LayersHiddenHidden = [2*numpy.random.random((self.hiddenNeutorn, self.hiddenNeutorn)) - 1 not usable in this problem
        #LayerHiddenOutput = [2*numpy.random.random((self.hiddenNeutorn, self.outputNeurons)) - 1 for i in range(self.nrHiddenLayers-1)]
        #LayerOutput = [Neuron(self.outputNeurons,1)]

        # create Neuron Layer
        self.Layers = [2*numpy.random.random((self.inputNeurons, self.outputNeurons)) - 1 for i in range(self.hiddenNeutorn)]
        #self.Layers += LayerInputHidden, LayerHiddenOutput
        #self.Layers = numpy.array(self.Layers)

    def sigmoid(self, information, i):
        net = numpy.dot(information, self.Layers[i])
        self.output = 1/(1+numpy.exp(-net))
        return self.output

    def derivate(self, x):
        return x*(1-x)

    def train(self):
        #for i
        output = self.sigmoid(self.trainingInputData, i)
n = Network(18, 2, 18, 1, [0,0,1])
#n.activateEachNeuron([0,0,1])
print(n.Layers)
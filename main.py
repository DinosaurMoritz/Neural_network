import math
import random

flatten = lambda self, t: [item for sublist in t for item in sublist]


class Neuron:
    def __init__(self):
        self.bias = 0
        self.weights = None

    def forward(self, inputs):
        lenInp = len(inputs)
        if not self.weights:
            self.weights = [random.uniform(-1, 1) for _ in range(lenInp)]

        output = [inputs[n] * self.weights[n] for n in range(lenInp)]

        self.s = sum(output) + self.bias
        self.output = self.sigmoid(self.s)
        return self.output


    @staticmethod
    def sigmoid(x):
        return 1 / (1 + math.exp(-x))

    def getWeights(self):
        return self.weights

    def setWeights(self, weights):
        self.weights = weights

    def weightEffect(self):
        # output = []
        # for w in self.weights:
        #     pass
        weight = self.weights[0]
        errorT = 0,5*()



class Layer:
    def __init__(self, numberOfNeurons, firstLayer=False):
        self.firstLayer = firstLayer
        self.numberOfNeurons = numberOfNeurons
        self.output = None
        self.neurons = [Neuron() for _ in range(self.numberOfNeurons)]

    def forward(self, inputs):
        if self.firstLayer:
            self.output = [n.forward([input]) for n, input in zip(self.neurons, inputs)]
        else:
            self.output = [n.forward(inputs) for n in self.neurons]
        return self.output

    def getWeights(self):
        return [n.getWeights() for n in self.neurons]

    def setWeights(self, weights):
        for neuronWeights, neuron in zip(weights, self.neurons):
            neuron.setWeights(neuronWeights)





class Network:
    def __init__(self, *args):
        self.args = args
        self.layers = [Layer(nN) for nN in args[1:]]
        self.layers.insert(0, Layer(args[0], True))
        self.forward([0 for _ in range(args[0])])

    def forward(self, inputs):
        for l in self.layers:
            inputs = l.forward(inputs)
        return inputs

    def getWeights(self):
        return [l.getWeights() for l in self.layers]

    def setWeights(self, weights):
        for layerWeights, layer in zip(weights, self.layers):
            layer.setWeights(layerWeights)

    def calcError(self, inputs, expected):
        outputList = self.forward(inputs)
        assert len(outputList) == len(expected), f"Got {len(inputs)} inputs and {len(expected)} expected values!"
        return sum([0.5*(target - output)**2 for target, output in zip(expected, outputList)])

    def back(self, inputs, expected):
        totalError = self.calcError(inputs, expected)
        pre = self.forward(inputs)
        n = self.layers[-1].neurons[0]
        w = n.weights[0]
        out1 = 0.5*pow((expected[0] - pre[0]), 2)
        return (out1)

    def getLayerOutput(self):
        return [l.output for l in self.layers]



if __name__ == "__main__":
    inp = [1, 2, 3]
    expected = [1,2]
    preWeights = [[[0.6642659387092582], [-0.5170373023517503], [0.017463164755874727]], [[-0.3469857033972712, 0.7309619431731142, 0.1977135007914237], [-0.024567308529295895, -0.8552789697152876, 0.229235930891297], [0.33896971081053384, 0.5505220318116801, -0.79772672264467], [0.24567504227966097, 0.14164021720839792, 0.6157445843253817]], [[0.8312610069393729, 0.09145610252084291, -0.8193817831310624, 0.949869646372971], [-0.07971653222607222, -0.564241532998655, -0.7477515766204943, 0.421978086372391]]]
    # n = Neuron()
    # r = n.get_weights()
    n = Network(3, 4, 2)
    n.setWeights(preWeights)

    r = n.back(inp, expected)
    #r = n.calcError(inp, expected)
    #r = n.
    """
    #n = Neuron()
    r = n.forward(inp)
    r = n.setWeights([-0.9148957847869212, 0.37777298814088134, 0.8166065406039196])
    """

    #r = n.weightEffect()

    print(r)

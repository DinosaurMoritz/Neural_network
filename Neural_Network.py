# from Neuron import Neuron
from resources import *
import time
import random
import math
import json


# random.seed(1)


class Neural_network:
    LEARNING_RATE = .1

    def __init__(self, *args, activation_bundle=bundled_sigmoid):
        if not args:
            raise ValueError("Specify input, hidden and output Layer sizes!")
        self.args = args
        hidden_neuron_numbers = [args[n + 1] for n in range(len(args) - 1)]

        self.hidden_layers = []
        num_inputs = args[0]
        self.num_inputs = num_inputs
        # live_activation_bundle = bundled_relu
        for num_of_neurons in hidden_neuron_numbers:
            self.hidden_layers.append(Neuron_layer(num_of_neurons, num_inputs, activation_bundle))
            # live_activation_bundle = activation_bundle
            num_inputs = num_of_neurons

        self.number_of_hidden_layers = len(self.hidden_layers) - 1

        self.forward = self.feed_forward

    def feed_forward(self, inputs):
        if len(inputs) != self.num_inputs:
            raise ValueError(
                "Number of inputs to feed forward function does not match number of inputs set in network initialisation!")

        for layer in self.hidden_layers:
            inputs = layer.feed_forward(inputs)
        return inputs

    def get_weights(self):
        net = []
        for layer in self.hidden_layers:
            layer_weighs = []
            for neuron in layer.neurons:
                layer_weighs.append(neuron.weights)
            net.append(layer_weighs)
        return net

    def set_weights(self, data):
        for layer, layer_data in zip(self.hidden_layers, data):
            for neuron, neuron_data in zip(layer.neurons, layer_data):
                neuron.weights = neuron_data

    def load(self, file_name="Neural_network"):
        with open(file_name + ".json", "r") as f:
            self.set_weights(json.load(f))

    def save(self, file_name="Neural_network"):
        with open(file_name + ".json", "w") as f:
            json.dump(self.get_weights(), f)

    # Uses online learning, ie updating the weights after each training case
    def train_once(self, training_inputs, training_outputs):
        self.feed_forward(training_inputs)

        all_errors_wrt_total_net = [[]]

        # OUTPUT NEURON DELTAS
        for training_output, neuron in zip(training_outputs, self.hidden_layers[-1].neurons):
            all_errors_wrt_total_net[0].append(neuron.calculate_pd_error_wrt_total_net_input(training_output))

        # HIDDEN NEURON DELTAS
        for hidden_layer_number in range(self.number_of_hidden_layers):
            hidden_layer = self.hidden_layers[-(hidden_layer_number + 2)]
            num_of_neurons_in_layer = len(hidden_layer.neurons)
            all_errors_wrt_total_net.append([0] * num_of_neurons_in_layer)
            for h in range(num_of_neurons_in_layer):
                d_error_wrt_hidden_neuron_output = 0
                layer_b4 = self.hidden_layers[-(hidden_layer_number + 1)]
                for o in range(len(layer_b4.neurons)):  # LAYER B4 CURRENT HIDDEN
                    d_error_wrt_hidden_neuron_output += all_errors_wrt_total_net[-2][o] * layer_b4.neurons[o].weights[h]

                all_errors_wrt_total_net[-1][h] = d_error_wrt_hidden_neuron_output * \
                                                  hidden_layer.neurons[h].d_activation_function(
                                                      hidden_layer.neurons[h].output)

        # UPDATE NEURON WEIGHTS
        for hidden_layer_number in range(self.number_of_hidden_layers + 1):
            hidden_layer = self.hidden_layers[-(hidden_layer_number + 1)]
            # HIDDENLAYER 0 --> DELTAS 0
            for neuron_number in range(len(hidden_layer.neurons)):
                for weight_number in range(len(hidden_layer.neurons[neuron_number].weights) - 1):
                    pd_error_wrt_weight = all_errors_wrt_total_net[hidden_layer_number][neuron_number] * \
                                          hidden_layer.neurons[neuron_number].inputs[weight_number]

                    hidden_layer.neurons[neuron_number].weights[weight_number] = \
                        hidden_layer.neurons[neuron_number].weights[weight_number] - (
                                self.LEARNING_RATE * pd_error_wrt_weight)

                # BIAS
                pd_error_wrt_weight = all_errors_wrt_total_net[hidden_layer_number][neuron_number]

                hidden_layer.neurons[neuron_number].weights[-1] = \
                    hidden_layer.neurons[neuron_number].weights[-1] - (
                            self.LEARNING_RATE * pd_error_wrt_weight)

    def _train_batch(self, batch):
        all_errors_wrt_total_net = [[0 for _ in layer.neurons] for layer in self.hidden_layers[::-1]]

        for training_inputs, training_outputs in batch:
            self.feed_forward(training_inputs)

            # OUTPUT NEURON DELTAS
            for neuron_number, (training_output, neuron) in enumerate(zip(training_outputs, self.hidden_layers[-1].neurons)):
                all_errors_wrt_total_net[0][neuron_number] += neuron.calculate_pd_error_wrt_total_net_input(training_output)

            # HIDDEN NEURON DELTAS
            for hidden_layer_number in range(self.number_of_hidden_layers):
                hidden_layer = self.hidden_layers[-(hidden_layer_number + 2)]
                num_of_neurons_in_layer = len(hidden_layer.neurons)
                for h in range(num_of_neurons_in_layer):
                    d_error_wrt_hidden_neuron_output = 0
                    layer_b4 = self.hidden_layers[-(hidden_layer_number + 1)]
                    for o in range(len(layer_b4.neurons)):  # LAYER B4 CURRENT HIDDEN
                        d_error_wrt_hidden_neuron_output += all_errors_wrt_total_net[hidden_layer_number][o] * \
                                                            layer_b4.neurons[o].weights[h]

                    all_errors_wrt_total_net[hidden_layer_number+1][h] = d_error_wrt_hidden_neuron_output * \
                                                      hidden_layer.neurons[h].d_activation_function(
                                                          hidden_layer.neurons[h].output)

        for hidden_layer_number in range(self.number_of_hidden_layers + 1):
            hidden_layer = self.hidden_layers[-(hidden_layer_number + 1)]
            # HIDDENLAYER 0 --> DELTAS 0
            for neuron_number in range(len(hidden_layer.neurons)):
                for weight_number in range(len(hidden_layer.neurons[neuron_number].weights) - 1):
                    pd_error_wrt_weight = all_errors_wrt_total_net[hidden_layer_number][neuron_number] * \
                                          hidden_layer.neurons[neuron_number].inputs[weight_number]

                    hidden_layer.neurons[neuron_number].weights[weight_number] = \
                        hidden_layer.neurons[neuron_number].weights[weight_number] - (
                                self.LEARNING_RATE * pd_error_wrt_weight)

                # BIAS
                pd_error_wrt_weight = all_errors_wrt_total_net[hidden_layer_number][neuron_number]

                hidden_layer.neurons[neuron_number].weights[-1] = \
                    hidden_layer.neurons[neuron_number].weights[-1] - (
                            self.LEARNING_RATE * pd_error_wrt_weight)


    def train(self, t_set, iterations=1000):
        print("Starting training!")
        t = time.time()
        for _ in range(iterations):
            for example in t_set:
                self.train_once(*example)
        print("Training took ", time.time() - t, "s!")

    def train_function(self, t_set, iterations=1000):
        print("Starting training!")
        t = time.time()
        for _ in range(iterations):
            for example in t_set():
                self.train_once(*example)
        print("Training took ", time.time() - t, "s!")

    def train_batch(self, t_set, iterations=1000):
        print("Starting training!")
        t = time.time()
        for _ in range(iterations):
            for ex in t_set():
                self._train_batch([ex])
        print("Training took ", time.time() - t, "s!")




class Neuron_layer:
    def __init__(self, num_neurons, number_of_inputs, activation_bundle):
        self.neurons = []
        for _ in range(num_neurons):
            self.neurons.append(Neuron(number_of_inputs, activation_bundle))

    def feed_forward(self, inputs):
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.calculate_output(inputs))
        return outputs


class Neuron:
    def __init__(self, number_of_inputs, activation_bundle=None):
        self.weights = [random.random() - 0.5 for _ in range(number_of_inputs + 1)]

        activation_bundle = activation_bundle if activation_bundle is not None else [self.sigmoid, self.d_sigmoid]

        self.activation_function, self.d_activation_function = activation_bundle

    def calculate_output(self, inputs) -> float:
        self.inputs = inputs
        total = 0
        for i in range(len(self.inputs)):
            total += self.inputs[i] * self.weights[i]
        self.output = self.activation_function(total + self.weights[-1])
        return self.output

    def sigmoid(self, inp) -> float:
        try:
            return 1 / (1 + pow(2.71828, -inp))
        except OverflowError:
            print("Sigmoid input to big!")
            return 1 if inp > 0 else 0

    def d_sigmoid(self, inp) -> float:
        return inp * (1 - inp)

    def calculate_pd_error_wrt_total_net_input(self, target_output) -> float:
        diriv = self.d_activation_function(self.output)
        return -(target_output - self.output) * diriv

    def calculate_error(self, target_output) -> float:
        return 0.5 * (target_output - self.output) ** 2


def get_mnist():
    with open("mnist_train.csv") as f:
        data = f.readlines()
        s = []
        empty_label = [0] * 10
        for example in data:
            sp = example.split(",")
            example = [int(n) / 255 for n in sp[1:]]
            label = empty_label[:]
            label[int(sp[0])] = 1
            s.append([example, label])
    return s


def get_average():
    ts = []
    for _ in range(100):
        pic = [random.random() for _ in range(100)]
        label = [sum(pic) / 100]
        ts.append([pic, label])
    return ts


"""
for pic, label in get_average():
    print(nn.forward(pic)[0], label[0])


actualx = [x / 10 for x in list(range(100))]
actualy = []
nety = []

for inp, label in get_average():
    r = nn.feed_forward(inp)[0]
    nety.append(r)
    actualy.append(label[0])

import matplotlib.pyplot as plt

actualy = sorted(actualy)
nety = sorted(nety)

plt.plot(actualx, actualy)
plt.plot(actualx, nety)
plt.title('Neural Net')

plt.savefig("Graph_.png", bbox_inches='tight')





def get_training_set():
    training_set = []
    for _ in range(100):
        r1 = random.random()
        r2 = random.random()
        if random.random() > 0.5:
            r3 = random.random()
            t = 0
        else:
            r3 = r2 + r1
            t = 1
        training_set.append([[r1, r2, r3], [t, 1 - t]])
    return training_set
"""


# nn = Neural_network(1, 5, 5, 1)


def get_training_set2():
    training_set = []
    for _ in range(100):
        r = round(random.randint(0, 628) / 100, 3)
        training_set.append([[r], [(math.sin(r) + 1) / 2]])
    return training_set


nn = Neural_network(1, 4, 1, activation_bundle=bundled_sigmoid)

nn.load()

#nn.train_batch(get_training_set2, 5000)  # Training took  8.930026054382324 s!


wrong = 0
actualx = [x / 100 for x in list(range(628))]
actualy = [round((math.sin(x) + 1) / 2, 3) for x in actualx]

nety = []

for inp in actualx:
    r = nn.feed_forward([inp])[0]
    nety.append(r)

import matplotlib.pyplot as plt

plt.plot(actualx, actualy)
plt.plot(actualx, nety)
plt.title('Neural Net')

plt.savefig("Graph1.png", bbox_inches='tight')

nn.save()

"""

import time

nn.load()

#nn.train(get_training_set2(), 5000)

# nn.save()

actualx = [x / 100 for x in list(range(628*2))]
actualy = [round((math.sin(x) + 1) / 2, 3) for x in actualx]

nety = []

for inp in actualx:
    r = nn.feed_forward([inp])[0]
    nety.append(r)

import matplotlib.pyplot as plt



diff = [0.5*(a-b)**2 * 100 for a,b in zip(actualy, nety)]

plt.plot(actualx, diff)


line = [0.023 for _ in diff]

plt.plot(actualx, line)


improvement_set = []

for i, err in enumerate(diff):
    if err > 0.002:
        improvement_set.append([[i/100], [(math.sin(i/100) + 1) / 2]])

random.shuffle(improvement_set)

nn.train(improvement_set, 100)



nety = []

for inp in actualx:
    r = nn.feed_forward([inp])[0]
    nety.append(r)


plt.plot(actualx, actualy)
plt.plot(actualx, nety)


plt.title('Neural Net')

plt.savefig("Graph3.png", bbox_inches='tight')

nn.save()
"""

"""

for inp, exp in get_training_set2():
    res = nn.forward(inp)
    print(res, exp)


right = 0

for ex in get_training_set():
    inp, exp = ex
    result = nn.feed_forward(inp)
    # print(result.index(max(result)), exp.index(1))
    if round(result[0]) == exp[0] and round(result[1]) == exp[1]:
        right += 1
        print(True)
    else:
        print(result, exp)
    # print(result, exp)

# You had correct :     0.98 / Training took  8.873940229415894 s!

print("You had correct :    ", right / len(get_training_set()))


"""

#
# print("You had correct :    ", right / len(get_training_set()))
#
#
# def graph_net(name="graph1"):
#     t = time.time()
#     res_list = []
#     for n in range(100):
#         r = nn.feed_forward([n / 100 * 7])[0]
#         res_list.append(r)
#
#     # importing the required module
#     import matplotlib.pyplot as plt
#
#     # x axis values
#     # x axis values
#     x1 = [x / 100 * 7 for x in range(100)]
#     # corresponding y axis values
#     y1 = [(math.sin(x / 100 * 7) + 1) / 2 for x in range(100)]
#
#     x2 = [x / 100 * 7 for x in range(100)]
#     y2 = [x for x in res_list]
#
#     # plotting the points
#     plt.plot(x1, y1)
#     plt.plot(x2, y2)
#     # naming the x axis
#     plt.xlabel('x - axis')
#     # naming the y axis
#     plt.ylabel('y - axis')
#
#     # giving a title to my graph
#     plt.title('Neural Net')
#
#     # function to show the plot
#     plt.savefig("Graph_" + name + ".png", bbox_inches='tight')
#     print(time.time() - t)


# graph_net()

# Training took  61.89749312400818 s! 0.887
# (3, 7, 1) Training took  16.102587699890137 s! 0.98 with 100 ex and 5000 it
# (3, 7, 4, 1) Training took  27.945143222808838 s! 0.99 with 100 ex and 5000 it
# [-10.329287003624671, 10.47732360684744, 5.108353072889753, 4.986094581962879, -10.38095320025427, -10.009561430376943, 9.971423515134363]

"""
class NeuralNet:
    def __init__(self, amount_of_inputs=0, size=None):
        self.size = size if size is not None else []

        self.learning_rate = 0.001

        self.number_of_layers = len(self.size)

        self.neurons = []
        self.input_amount = amount_of_inputs
        self.input_amount_for_loop = self.input_amount

        for layer_size in self.size:
            self.neurons.append([Neuron(self.input_amount_for_loop) for _ in range(layer_size)])
            self.input_amount_for_loop = len(self.neurons[-1])

    def __repr__(self):
        return f"NeuralNet {id(self)} with {self.input_amount} inputs and {self.size} Neurons!"

    def get_weights(self):
        return [[neuron.weights for neuron in layer] for layer in self.neurons]

    def set_weights(self, weights):
        for layer_number, layer_weights in enumerate(weights):
            for neuron_number, neuron_weights in enumerate(layer_weights):
                self.neurons[layer_number][neuron_number].weights = neuron_weights

    def forward(self, inputs, function=sigmoid):
        for layer in self.neurons:
            new_inputs = []
            for neuron in layer:
                new_inputs.append(neuron.forward(inputs, function))

            inputs = new_inputs
        return inputs

    def errors(self, results, targets):
        assert len(results) == len(targets)
        errors = [pow(target - result, 2) for target, result in zip(targets, results)]
        return sum(errors) / len(errors), errors

    def backward(self, inputs, targets):

        outputs = self.forward(inputs)
        total_error, errors = self.errors(outputs, targets)

        dA_dW1 = inputs[0]
        dA_dW2 = inputs[1]

        dC_dA = 2 * (outputs[0] - targets[0])

        dW1 = self.learning_rate * dA_dW1 * dC_dA
        dW2 = self.learning_rate * dA_dW2 * dC_dA

        return -dW1, -dW2

        self.neurons[-1][0].weights[0] -= dW1
        self.neurons[-1][0].weights[1] -= dW2

        print("total_error", total_error)

    def train(self, training_set):
        for training_inputs, expected_outputs in training_set:
            result = self.forward(training_inputs)
            has_more_than_one_layer = len(self.neurons) > 1

            # ------------------- OUTPUT ERROR -----------------------------

            output_errors = [pow(prediction - target, 2) for prediction, target in zip(result, expected_outputs)]


            # ------------------- HIDDEN ERROR ------------------------------
            hidden_errors = [0] * len(self.neurons[0])
            for h, hidden_neuron in enumerate(self.neurons[0]):
                # dE/dyⱼ = Σ ∂E/∂zⱼ * ∂z/∂yⱼ = Σ ∂E/∂zⱼ * wᵢⱼ
                d_error_wrt_hidden_neuron_output = sum(
                    [output_errors[o] * output_neuron.weights[h] for o, output_neuron in enumerate(self.neurons[1])])

                # ∂E/∂zⱼ = dE/dyⱼ * ∂zⱼ/∂
                hidden_errors[h] = d_error_wrt_hidden_neuron_output * hidden_neuron.output * (1 - hidden_neuron.output)

            # ------------------- CHANGE OUTPUT -----------------------------

            for o, neuron in enumerate(self.neurons[-1]):
                for w_ho in range(len(neuron.weights) - 1):
                    # Δw = α * ∂E/∂zⱼ * ∂zⱼ/∂wᵢⱼ
                    neuron.weights[w_ho] -= self.learning_rate * neuron.inputs[w_ho] * output_errors[o]
                    print(self.learning_rate * neuron.inputs[w_ho] * output_errors[o])

            # ------------------- CHANGE HIDDEN -----------------------------

            for h, neuron in enumerate(self.neurons[0]):
                for w_ih in range(len(self.neurons[0][h].weights) - 1):
                    # Δw = α * ∂E/∂zⱼ * ∂zⱼ/∂wᵢ
                    neuron.weights[w_ih] -= self.learning_rate * hidden_errors[h] * neuron.inputs[w_ih]


    def train1(self, training_inputs, training_outputs):
        self.feed_forward(training_inputs)

        # 1. Output neuron deltas
        pd_errors_wrt_output_neuron_total_net_input = [0] * len(self.output_layer.neurons)
        for o in range(len(self.output_layer.neurons)):
            # ∂E/∂zⱼ
            pd_errors_wrt_output_neuron_total_net_input[o] = self.output_layer.neurons[
                o].calculate_pd_error_wrt_total_net_input(training_outputs[o])

        # 2. Hidden neuron deltas
        pd_errors_wrt_hidden_neuron_total_net_input = [0] * len(self.hidden_layer.neurons)
        for h in range(len(self.hidden_layer.neurons)):

            # We need to calculate the derivative of the error with respect to the output of each hidden layer neuron
            # dE/dyⱼ = Σ ∂E/∂zⱼ * ∂z/∂yⱼ = Σ ∂E/∂zⱼ * wᵢⱼ
            d_error_wrt_hidden_neuron_output = 0
            for o in range(len(self.output_layer.neurons)):
                d_error_wrt_hidden_neuron_output += pd_errors_wrt_output_neuron_total_net_input[o] * \
                                                    self.output_layer.neurons[o].weights[h]

            # ∂E/∂zⱼ = dE/dyⱼ * ∂zⱼ/∂
            pd_errors_wrt_hidden_neuron_total_net_input[h] = d_error_wrt_hidden_neuron_output * \
                                                             self.hidden_layer.neurons[
                                                                 h].calculate_pd_total_net_input_wrt_input()

        # 3. Update output neuron weights
        for o in range(len(self.output_layer.neurons)):
            for w_ho in range(len(self.output_layer.neurons[o].weights)):
                # ∂Eⱼ/∂wᵢⱼ = ∂E/∂zⱼ * ∂zⱼ/∂wᵢⱼ
                pd_error_wrt_weight = pd_errors_wrt_output_neuron_total_net_input[o] * self.output_layer.neurons[
                    o].calculate_pd_total_net_input_wrt_weight(w_ho)

                # Δw = α * ∂Eⱼ/∂wᵢ
                self.output_layer.neurons[o].weights[w_ho] -= self.LEARNING_RATE * pd_error_wrt_weight

        # 4. Update hidden neuron weights
        for h in range(len(self.hidden_layer.neurons)):
            for w_ih in range(len(self.hidden_layer.neurons[h].weights)):
                # ∂Eⱼ/∂wᵢ = ∂E/∂zⱼ * ∂zⱼ/∂wᵢ
                pd_error_wrt_weight = pd_errors_wrt_hidden_neuron_total_net_input[h] * self.hidden_layer.neurons[
                    h].calculate_pd_total_net_input_wrt_weight(w_ih)

                # Δw = α * ∂Eⱼ/∂wᵢ
                self.hidden_layer.neurons[h].weights[w_ih] -= self.LEARNING_RATE * pd_error_wrt_weight


    def train(self, training_set, iterations=1000):
        for _ in range(iterations):
            w1 = 0
            w2 = 0
            for example in get_training_set():
                dW1, dW2 = net.backward(*example)
                w1 += dW1
                w2 += dW2
            net.neurons[-1][0].weights[0] += w1 / len(get_training_set())
            net.neurons[-1][0].weights[1] += w2 / len(get_training_set())



    def backward(self, inputs, targets):

        outputs = self.forward(inputs)
        output1 = self.neurons[-1][0].activation
        # output2 = self.neurons[-2][0].activation

        total_error, errors = self.errors(outputs, targets)

        dC_dA = 2 * (output1 - targets[0])

        dA_dY1 = output1 * (1 - output1)

        # dA_dY2 = output2 * (1 - output2)

        dW1 = self.learning_rate * dC_dA * dA_dY1

        # dW2 = dA_dY2 * dW1

        self.neurons[-1][0].weights[0] -= dW1
        # self.neurons[-1][0].weights[0] -= dW1

        print("total_error", total_error)


    def backward(self, inputs, targets):
        outputs = self.forward(inputs)

        assert len(outputs) == len(targets)
        errors = [0.5 * pow(target - result, 2) for target, result in zip(targets, outputs)]
        total_error, errors = sum(errors) / len(errors), errors

        # dErrorTotal/dOutput1
        totalErrors_outputs = [(target - output) for target, output in
                               zip(targets, outputs)]  # !!!!!!!!!!!!!! "-" in front of "(target-output)" !!!!!!!

        # dOutY/dY
        dOutY_dY = [output * (1 - output) for output in outputs]

        prev_outputs = [neuron.activation for neuron in self.neurons[-2]]

        dW5 = totalErrors_outputs[0] * dOutY_dY[0] * prev_outputs[0]

        print


    def backward(self, w, l, l_1):
        inputs, expected_outputs = example

        # FORWARD - START
        _inputs = inputs[:]
        all_outputs = [_inputs]
        for layer in self.neurons:
            new_inputs = []
            for neuron in layer:
                new_inputs.append(neuron.forward(_inputs, tanh))

            _inputs = new_inputs
            all_outputs.append(_inputs)
        actual_output = _inputs
        # FORWARD - END

        cost = pow(actual_output[0] - expected_outputs[0], 2)

        dC__A = 2 * (l.activation - expected_outputs[0])
        dA__dZ = d_sigmoid(l.output)
        dZ__dW = l_1.ativation

        dC__dW = dC__A * dA__dZ * dZ__dW
        dC__db = dC__A * dA__dZ


if __name__ == "__main__":
    amount_of_inputs = 2
    size = [1]

    net = NeuralNet(amount_of_inputs, size)

    net.train(get_training_set())  # backward([[5], [0, 5]])  # train(get_training_set(), 100)

    for example in get_training_set():
        print(example, net.forward(example[0], tanh)[0])

    # print("Execution took: ", time.time() - t, "s")
"""

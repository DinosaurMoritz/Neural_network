from Neuron import Neuron
from resources import *
import time
import random

random.seed(1)

import random
import math


class Neural_network:
    LEARNING_RATE = 0.2

    def __init__(self, *args):
        self.args = args
        num_inputs = args[0]
        # num_outputs = args[-1]
        self.num_inputs = num_inputs
        hidden_neuron_numbers = [args[n + 1] for n in range(len(args) - 1)]

        self.hidden_layers = []

        for num_of_neurons in hidden_neuron_numbers:
            self.hidden_layers.append(Neuron_layer(num_of_neurons, num_inputs))
            num_inputs = num_of_neurons

        self.number_of_hidden_layers = len(self.hidden_layers) - 1

    def feed_forward(self, inputs):
        for layer in self.hidden_layers:
            inputs = layer.feed_forward(inputs)
        return inputs  # self.output_layer.feed_forward(self.hidden_layer2.feed_forward(self.hidden_layer1.feed_forward(inputs)))

    def get_weights(self):
        return [[[neuron.weights, neuron.bias] for neuron in layer.neurons] for layer in self.hidden_layers]

    def save_net(self, file_name="Neural_network"):
        with open(file_name + ".txt", "a") as f:
            f.write(str(self.get_weights()) + "\n")

    def set_weights(self, data):
        for layer, neuron_data in zip(self.hidden_layers, data):
            for neuron, single_data in zip(layer.neurons, neuron_data):
                weights, bias = single_data
                neuron.weights = weights
                neuron.bias = bias

    # Uses online learning, ie updating the weights after each training case
    def train_once(self, training_inputs, training_outputs):
        self.feed_forward(training_inputs)

        all_pd_errors_wrt_hidden_neuron_total_net_input = [[0] * self.args[-1]]

        # OUTPUT NEURON DELTAS
        for o in range(self.args[-1]):
            # ∂E/∂zⱼ
            all_pd_errors_wrt_hidden_neuron_total_net_input[-1][o] = self.hidden_layers[-1].neurons[
                o].calculate_pd_error_wrt_total_net_input(training_outputs[o])

        # HIDDEN NEURON DELTAS
        for hidden_layer_number in range(self.number_of_hidden_layers):
            # hidden_layer_number + 1
            hidden_layer = self.hidden_layers[-(hidden_layer_number + 1)]
            all_pd_errors_wrt_hidden_neuron_total_net_input.append([0] * len(hidden_layer.neurons))
            for h in range(len(hidden_layer.neurons)):
                d_error_wrt_hidden_neuron_output = 0
                for o in range(len(hidden_layer.neurons)):
                    d_error_wrt_hidden_neuron_output += all_pd_errors_wrt_hidden_neuron_total_net_input[-1][o] * \
                                                        hidden_layer.neurons[o].weights[h]

                # ∂E/∂zⱼ = dE/dyⱼ * ∂zⱼ/∂
                delta = d_error_wrt_hidden_neuron_output * hidden_layer.neurons[h].d_sigmoid()
                #print(delta)
                all_pd_errors_wrt_hidden_neuron_total_net_input[-1][h] = d_error_wrt_hidden_neuron_output * \
                                                                         hidden_layer.neurons[h].d_sigmoid()

        """
        # 2. Hidden neuron deltas
        pd_errors_wrt_hidden2_neuron_total_net_input = [0] * len(self.hidden_layer2.neurons)
        for h in range(len(self.hidden_layer2.neurons)):
            d_error_wrt_hidden2_neuron_output = 0
            for o in range(len(self.output_layer.neurons)):
                d_error_wrt_hidden2_neuron_output += pd_errors_wrt_output_neuron_total_net_input[o] * \
                                                     self.output_layer.neurons[o].weights[h]

            # ∂E/∂zⱼ = dE/dyⱼ * ∂zⱼ/∂
            pd_errors_wrt_hidden2_neuron_total_net_input[h] = d_error_wrt_hidden2_neuron_output * \
                                                              self.hidden_layer2.neurons[
                                                                  h].d_sigmoid()

        # 2.5 Hidden neuron deltas
        pd_errors_wrt_hidden1_neuron_total_net_input = [0] * len(self.hidden_layer1.neurons)
        for h in range(len(self.hidden_layer1.neurons)):

            # We need to calculate the derivative of the error with respect to the output of each hidden layer neuron
            # dE/dyⱼ = Σ ∂E/∂zⱼ * ∂z/∂yⱼ = Σ ∂E/∂zⱼ * wᵢⱼ
            d_error_wrt_hidden1_neuron_output = 0
            for o in range(len(self.hidden_layer2.neurons)):
                d_error_wrt_hidden2_neuron_output += pd_errors_wrt_hidden2_neuron_total_net_input[o] * \
                                                     self.hidden_layer2.neurons[o].weights[h]

            # ∂E/∂zⱼ = dE/dyⱼ * ∂zⱼ/∂
            pd_errors_wrt_hidden1_neuron_total_net_input[h] = d_error_wrt_hidden1_neuron_output * \
                                                              self.hidden_layer1.neurons[
                                                                  h].d_sigmoid()
        """
        """
        # UPDATE NEURON WEIGHTS
        for hidden_layer_number in range(self.number_of_hidden_layers+1):
            hidden_layer = hidden_layer = self.hidden_layers[-(hidden_layer_number+1)]

            for o in range(len(hidden_layer.neurons)):
                for w_ho in range(len(hidden_layer.neurons[o].weights)):
                    # ∂Eⱼ/∂wᵢⱼ = ∂E/∂zⱼ * ∂zⱼ/∂wᵢⱼ
                    p1 = all_pd_errors_wrt_hidden_neuron_total_net_input
                    p2 = p1[-(hidden_layer_number + 1)]
                    p3 = p2[o]
                    p4 = hidden_layer.neurons[o].inputs[w_ho]
                    pd_error_wrt_weight = p3 * p4

                    # Δw = α * ∂Eⱼ/∂wᵢ
                    hidden_layer.neurons[o].weights[w_ho] -= self.LEARNING_RATE * pd_error_wrt_weight
        """

        # UPDATE NEURON WEIGHTS
        for hidden_layer_number in range(self.number_of_hidden_layers):  # range(self.number_of_hidden_layers + 1)
            hidden_layer = self.hidden_layers[-(hidden_layer_number + 1)]
            # HIDDENLAYER 0 --> DELTAS 0
            for neuron_number in range(len(hidden_layer.neurons)):
                for weight_number in range(len(hidden_layer.neurons[neuron_number].weights)):
                    # ∂Eⱼ/∂wᵢⱼ = ∂E/∂zⱼ * ∂zⱼ/∂wᵢⱼ
                    p1 = all_pd_errors_wrt_hidden_neuron_total_net_input[hidden_layer_number][neuron_number]

                    p4 = hidden_layer.neurons[neuron_number].inputs[weight_number]
                    pd_error_wrt_weight = p1 * p4

                    # Δw = α * ∂Eⱼ/∂wᵢ
                    hidden_layer.neurons[neuron_number].weights[
                        weight_number] -= self.LEARNING_RATE * pd_error_wrt_weight
        """
        # 4. Update hidden neuron weights
        for h in range(len(self.hidden_layer2.neurons)):
            for w_ih in range(len(self.hidden_layer2.neurons[h].weights)):
                # ∂Eⱼ/∂wᵢ = ∂E/∂zⱼ * ∂zⱼ/∂wᵢ
                pd_error_wrt_weight = pd_errors_wrt_hidden2_neuron_total_net_input[h] * self.hidden_layer2.neurons[h].inputs[w_ih]

                # Δw = α * ∂Eⱼ/∂wᵢ
                self.hidden_layer2.neurons[h].weights[w_ih] -= self.LEARNING_RATE * pd_error_wrt_weight

        
        # 3. Update output neuron weights
        for o in range(len(self.output_layer.neurons)):
            for w_ho in range(len(self.output_layer.neurons[o].weights)):
                # ∂Eⱼ/∂wᵢⱼ = ∂E/∂zⱼ * ∂zⱼ/∂wᵢⱼ
                pd_error_wrt_weight = pd_errors_wrt_output_neuron_total_net_input[o] * self.output_layer.neurons[
                    o].inputs[w_ho]

                # Δw = α * ∂Eⱼ/∂wᵢ
                self.output_layer.neurons[o].weights[w_ho] -= self.LEARNING_RATE * pd_error_wrt_weight

        # 4. Update hidden neuron weights
        for h in range(len(self.hidden_layer2.neurons)):
            for w_ih in range(len(self.hidden_layer2.neurons[h].weights)):
                # ∂Eⱼ/∂wᵢ = ∂E/∂zⱼ * ∂zⱼ/∂wᵢ
                pd_error_wrt_weight = pd_errors_wrt_hidden2_neuron_total_net_input[h] * self.hidden_layer2.neurons[
                    h].inputs[w_ih]

                # Δw = α * ∂Eⱼ/∂wᵢ
                self.hidden_layer2.neurons[h].weights[w_ih] -= self.LEARNING_RATE * pd_error_wrt_weight

        # 4.5 Update hidden neuron weights
        for h in range(len(self.hidden_layer1.neurons)):
            for w_ih in range(len(self.hidden_layer1.neurons[h].weights)):
                # ∂Eⱼ/∂wᵢ = ∂E/∂zⱼ * ∂zⱼ/∂wᵢ
                pd_error_wrt_weight = pd_errors_wrt_hidden1_neuron_total_net_input[h] * \
                                      self.hidden_layer1.neurons[h].inputs[w_ih]

                # Δw = α * ∂Eⱼ/∂wᵢ
                self.hidden_layer1.neurons[h].weights[w_ih] -= self.LEARNING_RATE * pd_error_wrt_weight
        """

    def train(self, t_set, iterations=1000):
        t = time.time()
        for _ in range(iterations):
            for example in t_set:
                self.train_once(*example)
        print("Training took ", time.time() - t, "s!")


class Neuron_layer:
    def __init__(self, num_neurons, number_of_inputs):

        # Every neuron in a layer shares the same bias
        self.bias = random.random()

        self.neurons = []
        for _ in range(num_neurons):
            self.neurons.append(Neuron(number_of_inputs, self.bias))

    def feed_forward(self, inputs):
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.calculate_output(inputs))
        return outputs


class Neuron:
    def __init__(self, number_of_inputs, bias):
        self.bias = bias
        self.weights = [random.random() for _ in range(number_of_inputs)]

    def calculate_output(self, inputs):
        self.inputs = inputs
        self.output = self.sigmoid(self.calculate_total_net_input())
        return self.output

    def calculate_total_net_input(self):
        total = 0
        for i in range(len(self.inputs)):
            total += self.inputs[i] * self.weights[i]
        return total + self.bias

    def sigmoid(self, total_net_input):
        return 1 / (1 + math.exp(-total_net_input))

    def calculate_pd_error_wrt_total_net_input(self, target_output):
        return -(target_output - self.output) * self.d_sigmoid();

    def calculate_error(self, target_output):
        return 0.5 * (target_output - self.output) ** 2

    def d_sigmoid(self):
        return self.output * (1 - self.output)


nn = Neural_network(3, 7, 1)


# nn = NeuralNetwork(3, 7, 1)


def get_training_set():
    training_set = []
    for _ in range(1000):
        r1 = random.random()
        r2 = random.random()
        if random.random() > 0.5:
            r3 = random.random()
            t = 0
        else:
            r3 = r2 + r1
            t = 1
        training_set.append([[r1, r2, r3], [t]])
    return training_set


def get_training_set2():
    training_set = []
    for _ in range(1000):
        r1 = random.random() * 100
        r2 = random.random() * 100
        r3 = random.random() * 100
        if r1 > r2 > r3:
            t = 1
        else:
            t = 0
        training_set.append([[r1, r2, r3], [t]])
    return training_set


# nn.set_weights([[[[0.8474337369372327, 0.763774618976614, 0.2550690257394217], 0.13436424411240122], [[0.49543508709194095, 0.4494910647887381, 0.651592972722763], 0.13436424411240122], [[0.7887233511355132, 0.0938595867742349, 0.02834747652200631], 0.13436424411240122], [[0.8357651039198697, 0.43276706790505337, 0.762280082457942], 0.13436424411240122], [[0.0021060533511106927, 0.4453871940548014, 0.7215400323407826], 0.13436424411240122], [[0.22876222127045265, 0.9452706955539223, 0.9014274576114836], 0.13436424411240122], [[0.030589983033553536, 0.0254458609934608, 0.5414124727934966], 0.13436424411240122]], [[[-139.43502635817362, 72.46007413446024, -62.911229959786525, 97.58693943124382, 35.04982978781019, -75.1804262169478, 73.31444180261545], 0.9391491627785106]]])#nn.train(get_training_set())
nn.train(get_training_set())

correct_guesses = 0
for ex in get_training_set():
    r = nn.feed_forward(ex[0])[0]
    correct = round(r) == round(ex[1][0])
    # print(correct, ex, r)
    if correct:
        correct_guesses += 1
    else:
        print(ex, ex[0][0] + ex[0][1], r)

print(correct_guesses / len(get_training_set()))

nn.save_net()
print()
# Training took  61.89749312400818 s! 0.887

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

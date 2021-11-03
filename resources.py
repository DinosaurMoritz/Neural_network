import math


e = 2.71828


def sigmoid(inp):
    try:
        return 1 / (1 + pow(e, -inp))
    except OverflowError:
        # print("Error with sigmoid input: ", inp)
        return 1 if inp > 0 else 0


def d_sigmoid(inp):
    return inp * (1 - inp)


def leaky_ReLU(inp):
    return inp if inp > 0 else 0.01 * inp


def d_leaky_ReLU(inp):
    return 1 if inp > 0 else 0.01


def tanh(inp):
    return (pow(e, inp) - pow(e, -inp)) / (pow(e, inp) + pow(e, -inp))


def d_tanh(inp):
    return 1 - ((tanh(inp)) ** 2)



bundled_sigmoid = [sigmoid, d_sigmoid]
bundled_relu = [leaky_ReLU, d_leaky_ReLU]
bundled_tanh = [tanh, d_tanh]


"""
actualx = [(x-5000) / 100 for x in list(range(10000))]
actualy = [sim(x) for x in actualx]
actualy2 = [d_sim(x) for x in actualx]


import matplotlib.pyplot as plt

plt.plot(actualx, actualy)
plt.plot(actualx, actualy2)

plt.savefig("Graph_.png", bbox_inches='tight')
"""


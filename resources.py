

def sigmoid(inp):
    try:
        return 1 / (1 + pow(2.71828, -inp))
    except OverflowError:
        # print("Error with sigmoid input: ", inp)
        return 1 if inp > 0 else 0

def d_sigmoid(inp):
    return inp * (1 - inp)


def leaky_ReLU(inp):
    return inp if inp > 0 else 0.01 * inp

def d_leaky_ReLU(inp):
    return 1 if inp > 0 else 0.01

"""
def tanh(inp):
    return 2*sigmoid(2*inp)-1

def d_tanh(inp):
    
"""

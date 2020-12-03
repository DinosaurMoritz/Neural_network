import numpy as np
import time
import random 

"""
class LayerDense:
    
    def __init__(self, nInputs, nNeurons):
        self.weights = self.randn(nInputs, nNeurons)
        self.biases = [0 for n in range(nNeurons)]
        
        self.nInputs = nInputs
        self.nNeurons = nNeurons
        
        
    def forward(self, inputs):
        self.output = self.addBiases(self.dot(inputs, self.weights), self.biases)
        
    def dot(self, a, b):
        c = [[0 for j in range(len(b[i]))] for i in range(len(a))]
        for i in range(len(c)):
            for j in range(len(c[i])):
                t = 0
                for k in range(len(b)):
                    t += a[i][k] * b[k][j]
                c[i][j] = t
        return c
        
    def randn(self, x, y):
        return [[random.uniform(-1,1) for n2 in range(y)] for n1 in range(x)]
    
    def addBiases(self, dp, bs):
        return [[z+bs[y] for z in dp[y]] for y in range(len(dp))]
            
            
o = [[1,2,3,2.5],
     [2,5,-1,2],
     [-1.5,2.7,3.3,-0.8]]       
        
      
        
l1 = LayerDense(4,5)
l2 = LayerDense(4,5)

l1.forward(o)
l2.forward(l1.output)
print(l2.output)
#ld.forward()
"""

class Layer:
    
    def __init__(self, nNeurons, nInputs):
        self.nNeurons = nNeurons
        self.nInputs = nInputs
        self.biases = [0 for n in range(nNeurons)]
        self.weights = [[random.uniform(-1,1) for i in range(nInputs)] for n in range(nNeurons)] 
    
    def forward(self, inputs):
        output = []
        for wa in self.weights:
            output.append([x * y for y in wa])
            
        ret = []
        for el in output:
            ret.append(sum(el))
        return output
        
    def printArr(self,arr):
        
        print("\n".join([str(ell) for ell in arr]))

        
inputs = [1,2,3,4,5]        
l1 = Layer(len(inputs),1)#neurons, inputs

r = l1.forward(inputs)
#l2 = Layer(len(r), 1)
#r = l2.forward(r)
print(r[0])

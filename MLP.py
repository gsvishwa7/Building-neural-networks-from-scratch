# Vinayagar
import numpy as np
import random

class MLP():
    def __init__(self, size_per_layer):
        self.size_per_layer = size_per_layer # e.g. [784,16,16,1]
        self.no_of_layers = len(size_per_layer)
        self.weights = [np.random.randn(size_per_layer[i-1], size_per_layer[i]) for i in range(1, self.no_of_layers)]
        self.biases = [np.random.randn(N,1) for N in size_per_layer[1:]]

    def feed_forward(self, x):
        for w, b in zip(self.weights, self.biases):
            x = sigmoid(np.matmul(w,x) + b)

def ReLU(x):
    """The Rectified Linear Unit (ReLU) function"""
    if x < 0:
        return 0
    else:
        return x
    
def ReLU_prime(x):
    """Derivative of the ReLU function"""
    if x<0:
        return 0
    else:
        return 1
    
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))
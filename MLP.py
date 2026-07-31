# Vinayagar
import numpy as np
import random
import mnist_loader

class MLP():
    def __init__(self, size_per_layer):
        self.size_per_layer = size_per_layer # e.g. [784,16,16,10]
        self.no_of_layers = len(size_per_layer)
        self.weights = [np.random.randn(size_per_layer[i], size_per_layer[i-1]) for i in range(1, self.no_of_layers)]
        self.biases = [np.random.randn(N,1) for N in size_per_layer[1:]]

    def feed_forward(self, x, return_inner_layers = True):
        if return_inner_layers == True:
            A = [x]
            Z = []
            for w, b in zip(self.weights, self.biases):
                z = np.matmul(w,x) + b
                Z.append(z)
                x = sigmoid(z)
                A.append(x)
            return Z, A # returns a list of Zs and activations per layer

        for w, b in zip(self.weights, self.biases):
            x = sigmoid(np.matmul(w,x) + b)
        return x

        
    
    def backprop(self, x, y):

        # Initialise gradients; each of these lists contains the gradient of the cost function
        # with respect to the weights or biases of each layer of the MLP
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # Initialise the zs and activations per layer
        z_per_layer, a_per_layer = self.feed_forward(x, return_inner_layers=True)

        # Set the values of the gradients for the last layer
        delta = (a_per_layer[-1] - y) * sigmoid_prime(z_per_layer[-1]) # A useful intermediary for calculations
        nabla_b[-1] = delta
        nabla_w[-1] = np.matmul(delta, a_per_layer[-2].transpose())


        


def ReLU(x):
    """The Rectified Linear Unit (ReLU) function"""
    return np.maximum(0, x)
    
def ReLU_prime(x):
    """Derivative of the ReLU function"""
    return np.heaviside(x, 1)
    
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))



training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
training_data = list(training_data)

### Some stuff to implement fully vectorised feed_forward() and backprop() methods
X = [x[0] for x in training_data] 
X_matrix = np.hstack(X)
print(X_matrix.shape)
print(np.array(X[0]).shape)

# test_NN = MLP([784,16,16,10])
# test_NN.feed_forward(x=training_data[0][0])

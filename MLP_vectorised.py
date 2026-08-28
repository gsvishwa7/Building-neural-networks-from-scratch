# Vinayagar
import numpy as np
import random
import mnist_loader

### Load data
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
training_data = list(training_data) # Contains 50000 tuples, each of the form (784 x 1, 10 x 1) array. 
                                    # So the first entry of each tuple is the image data and the second entry is the label
test_data = list(test_data)

### The key idea behind the fully vectorised feed_forward() and backprop() methods is to 
### change the shape of the inpute image data from 50000 x 784 x 1 to 784 x 50000.
### Each column of this reshaped training data corresponds to one input. In this manner,
### we are able to compute the action of weight matrices on each column individually as part
### of ordinary matrix multiplication. As a result, we are computing the activations
### corresponding to all the inputs at once, instead of having to loop over the inputs.

X = [x[0] for x in training_data]
print(f'{np.array(X).shape}, shape of each element: {np.array(X[0]).shape}')
X_matrix = np.hstack(X)
print(f'{X_matrix.shape}, shape of each element: {X_matrix[0].shape}')

def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

class MLP_enhanced():
    def __init__(self, size_per_layer):
        self.size_per_layer = size_per_layer # e.g. [784,16,16,10]
        self.no_of_layers = len(size_per_layer)
        self.weights = [np.random.randn(size_per_layer[i], size_per_layer[i-1]) for i in range(1, self.no_of_layers)]
        self.biases = [np.random.randn(N,1) for N in size_per_layer[1:]]

    def feed_forward(self, X, return_inner_layers = True):
        if return_inner_layers == True:
            A = [X] # Initialise input layer of activations, for every input in the mini-batch
            Z = [] # Initialise a list that will store z = wx + b 
            for w, b in zip(self.weights, self.biases):
                z = np.matmul(w,X) + b # Note: Numpy knows to broadcast b across all column
                Z.append(z)
                x = sigmoid(z)
                A.append(X)
            return Z, A # returns a list of Zs and activations per layer

        for w, b in zip(self.weights, self.biases):
            X = sigmoid(np.matmul(w,X) + b)
        return X

    def backprop(self, X, Y):
        # Initialise gradients; each element of these lists contains the gradient of the cost function
        # with respect to the weights/biases for each layer of the MLP, for all inputs of the mini-batch
        nabla_b = [np.zeros(b.shape) for b in self.biases] 
        nabla_w = [np.zeros(w.shape) for w in self.weights] 

        # Initialise the zs and activations per layer, for all inputs of the mini-batch
        z_per_layer, a_per_layer = self.feed_forward(X, return_inner_layers=True)

        # Set the values of the gradients for the last layer
        delta = (a_per_layer[-1] - Y) * sigmoid_prime(z_per_layer[-1]) # A useful intermediary for calculations
        nabla_b[-1] = delta
        nabla_w[-1] = np.matmul(delta, a_per_layer[-2].transpose())

        # Computing the gradient by going backwards layer by layer
        for l in range(2,self.no_of_layers):
            delta = np.matmul(self.weights[-l+1].transpose(),delta) * sigmoid_prime(z_per_layer[-l])
            nabla_b[-l] = delta
            nabla_w[-l] = np.outer(delta , a_per_layer[-l-1]) # Since we have explicit column vectors, delta and a_per_layer[-l-1], both with shapes (16,1), 
                                                          # we could have just said nabla_w = delta @ a_per_layer[-l-1].T. However, I wanted to make it
                                                          # clear that this is an outer product of vectors. 
        return nabla_b, nabla_w



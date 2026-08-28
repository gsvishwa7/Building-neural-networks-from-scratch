# Vinayagar
import numpy as np
import random
import mnist_loader


### The key idea behind the fully vectorised feed_forward() and backprop() methods is to 
### change the shape of the inpute image data from 50000 x 784 x 1 to 784 x 50000.
### Each column of this reshaped training data corresponds to one input. In this manner,
### we are able to compute the action of weight matrices on each column individually as part
### of ordinary matrix multiplication. As a result, we are computing the activations
### corresponding to all the inputs at once, instead of having to loop over the inputs.

# X = [x[0] for x in training_data]
# print(f'{np.array(X).shape}, shape of each element: {np.array(X[0]).shape}')
# X_matrix = np.hstack(X)
# print(f'{X_matrix.shape}, shape of each element: {X_matrix[0].shape}')
# Y = [y[1] for y in training_data]
# print(f'{np.array(Y).shape}, shape of each element: {np.array(Y[0]).shape}')
# Y_matrix = np.hstack(Y)
# print(f'{Y_matrix.shape}, shape of each element: {Y_matrix[0].shape}')

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
                X = sigmoid(z)
                A.append(X)
            return Z, A # returns a list of Zs and activations per layer

        for w, b in zip(self.weights, self.biases):
            X = sigmoid(np.matmul(w,X) + b)
        return X

    def backprop(self, X, Y):
            # Initialise gradients with the shapes of the weights and biases.
            nabla_b = [np.zeros(b.shape) for b in self.biases] 
            nabla_w = [np.zeros(w.shape) for w in self.weights] 

            # Initialise the zs and activations per layer, for all inputs of the mini-batch
            z_per_layer, a_per_layer = self.feed_forward(X, return_inner_layers=True)
            
            # Element-wise multiplication for the error
            delta = (a_per_layer[-1] - Y) * sigmoid_prime(z_per_layer[-1])
            
            # Sum across the batch dimension (axis=1) to get a column vector of shape (out_neurons, 1)
            nabla_b[-1] = np.sum(delta, axis=1, keepdims=True)
            
            # einsum 'ik,jk->ij' computes the outer product and sums over the batch dimension 'k'
            nabla_w[-1] = np.einsum('ik,jk->ij', delta, a_per_layer[-2])

            # Computing the gradient by going backwards layer by layer
            for l in range(2, self.no_of_layers):
                delta = np.matmul(self.weights[-l+1].transpose(), delta) * sigmoid_prime(z_per_layer[-l])
                
                # Again, sum across the batch dimension for biases
                nabla_b[-l] = np.sum(delta, axis=1, keepdims=True)
                
                # Again, let einsum sum across the batch dimension 'k' for weights
                nabla_w[-l] = np.einsum('ik,jk->ij', delta, a_per_layer[-l-1])
                
            return nabla_b, nabla_w

    def update_mini_batch(self, X, Y, eta):
        nabla_b, nabla_w = self.backprop(X, Y)
        change_in_biases = np.einsum('ij->i', nabla_b)
        change_in_weights = np.einsum('ijk->ij', nabla_w)
        self.weights = [w-(eta/X.shape[1])*nw for w, nw in zip(self.weights, change_in_weights)] 
        self.biases = [b-(eta/X.shape[1])*nb for b, nb in zip(self.biases, change_in_biases)] 
    
    def SGD(self, training_data, mini_batch_size, epochs, eta, test_data=None):
        """Train the neural network using stochastic gradient descent. 
        If ``test_data`` is provided then the network will be evaluated 
        against the test data after each epoch, and partial progress printed out.  
        This is useful for tracking progress, but slows things down substantially."""

        training_data = list(training_data) # The ``training_data`` is a list of 50000 tuples ``(x, y)`` representing the  training inputs and the desired outputs.
                                            # Each tuple contains (784x1,10x1) column vectors and there are 50000 such tuples.       
        n = len(training_data)

        # if test_data:
        #     test_data = list(test_data)
        #     n_test = len(test_data)

        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                X = np.hstack([x[0] for x in mini_batch])
                Y = np.hstack([y[1] for y in mini_batch])
                self.update_mini_batch(X, Y, eta)
            # if test_data:
            #     print(f"Epoch {j} : {self.evaluate(test_data)} / {n_test}")
            # else:
            #     print(f"Epoch {j+1} complete")
            print(f'Epoch {j+1} complete.')

### Load data
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
training_data = list(training_data) # Contains 50000 tuples, each of the form (784 x 1, 10 x 1) array. 
                                    # So the first entry of each tuple is the image data and the second entry is the label
test_data = list(test_data)

my_NN = MLP_enhanced([784,16,16,10])
my_NN.SGD(training_data, epochs = 10, mini_batch_size=200, eta=3)
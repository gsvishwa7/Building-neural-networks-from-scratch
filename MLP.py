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

        # Computing the gradient by going backwards layer by layer
        for l in range(2,self.no_of_layers):
            delta = np.matmul(self.weights[-l+1].transpose(),delta) * sigmoid_prime(z_per_layer[-l])
            nabla_b[-l] = delta
            nabla_w[-l] = np.outer(delta , a_per_layer[-l-1]) # Since we have explicit column vectors, delta and a_per_layer[-l-1], both with shapes (16,1), 
                                                          # we could have just said nabla_w = delta @ a_per_layer[-l-1].T. However, I wanted to make it
                                                          # clear that this is an outer product of vectors. 
        return nabla_b, nabla_w

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch: # For each input in the mini-batch, run the backprop() algorithm and the gradient 
            nabla_b_single_input, nabla_w_single_input = self.backprop(x, y) # The output of backprop() are the two gradient lists calculated from a single input
            nabla_b = [nb+delta_nb for nb, delta_nb in zip(nabla_b, nabla_b_single_input)] 
            nabla_w = [nw+delta_nw for nw, delta_nw in zip(nabla_w, nabla_w_single_input)] # Note that we are adding to the gradients but we still need to divide by
                                                                                           # len(mini_batch) since we take an average
        self.weights = [w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)] 
        self.biases = [b-(eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)] 
    
    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """Train the neural network using stochastic gradient descent. 
        If ``test_data`` is provided then the network will be evaluated 
        against the test data after each epoch, and partial progress printed out.  
        This is useful for tracking progress, but slows things down substantially."""

        training_data = list(training_data) # The ``training_data`` is a list of 50000 tuples ``(x, y)`` representing the  training inputs and the desired outputs.
                                            # Each tuple contains (784x1,10x1) column vectors and there are 50000 such tuples.       
        n = len(training_data)

        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)

        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print(f"Epoch {j} : {self.evaluate(test_data)} / {n_test}")
            else:
                print(f"Epoch {j+1} complete")

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        # Evaluate the neural network against test data
        # Output of the network is the node with the highest activation in the final layer
        test_results = [(np.argmax(self.feed_forward(x, return_inner_layers = False)), y) for (x, y) in test_data]

        # Return the number of test inputs for which the neural network outputs the correct result.
        return sum(int(x == y) for (x, y) in test_results)


### Useful functions
def ReLU(x):
    """The Rectified Linear Unit (ReLU) function"""
    return np.maximum(0, x)
    
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))


### Load data
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
training_data = list(training_data)
test_data = list(test_data)
### Initialise and train the neural network
my_first_neural_net = MLP([784,16,16,10])
my_first_neural_net.SGD(training_data, epochs = 50, mini_batch_size=20, eta=4)
print(f'{my_first_neural_net.evaluate(test_data)} / {len(test_data)} correct.')


# Y = [y[1] for y in training_data]
# print(np.array(Y).shape)

# test_NN = MLP([784,16,16,10])
# test_x, test_y = training_data[0]
# A, Z = test_NN.feed_forward(test_x, return_inner_layers=True)
# nabla_b, nabla_w = test_NN.backprop(test_x, test_y)
# for i in range(len(nabla_b)):
#     print(np.array(nabla_b[i]).shape)
#     print(np.array(nabla_w[i]).shape)
#     print(A[i].shape)
# print(A[-1].shape, A[-2].shape)





'''
A feed forward Neural Network which identifies Handwritten Digits
'''

# Import libraries
import numpy as np
import random

# Import Files
import load_data

"""
Helper functions
"""


# function meant to load biases from an npy file, used so that a new network doesn't have to be trained from scratch
def load_biases():
    saved_biases = np.load('biases.npy', allow_pickle=True)
    return saved_biases


# function meant to load weights from an npy file, used so that a new network doesn't have to be trained from scratch
def load_weights():
    saved_weights = np.load('weights.npy', allow_pickle=True)
    return saved_weights


# Sigmoid function
def sigmoid(z):
    # meant to smooth output and place all values between 0 and 1
    return 1 / (1 + np.exp(-z))


# First derivative of the sigmoid function
def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))


# Our Neural Network Class
class Network(object):

    # Initialize a neural network of given number of neurons per layer in parameter `sizes`
    # Initialize with random weights and biases
    def __init__(self, sizes, load_network=None):
        self.num_layers = len(sizes)
        self.sizes = sizes

        if load_network:  # if a network has already been trained, load saved biases and weights
            self.biases = load_biases()
            self.weights = load_weights()

        else:
            # No biases in the input layer
            self.biases = [np.random.randn(y, 1) for y in sizes[1:]]

            # Weights go between consecutive layers
            self.weights = [np.random.randn(y, x) for x, y in list(zip(sizes[:-1], sizes[1:]))]

    def feedfoward(self, x):
        """
        Returns the output of the network as x which is the 10-dimensional ndarray meant to represent the network's guess
        :param x: ndarray
            numpy array meant to be the activations
        :return: x
        """
        for b, w in list(zip(self.biases, self.weights)):
            x = sigmoid(np.dot(w, x) + b)

        return x

    # Stochastic Gradient Descent function
    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        # If Test Data is passed in, it will test run the neural network after every epoch, and
        # print out the results; the accuracy of the neural network in digit recognition
        if test_data:
            n_test = len(test_data)
            eval = self.evaluate(test_data)
            print(f"Epoch Untrained: {eval} / {n_test} with accuracy {round(float(eval) / n_test * 100, 2)}%")

        n_tr = len(training_data)

        for j in range(epochs):

            # Shuffles training data then segments it into N equal samples.
            random.shuffle(training_data)
            mini_batches = [training_data[k:k + mini_batch_size] for k in range(0, n_tr, mini_batch_size)]

            # Runs training on every batch
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)

            if test_data:
                eval = self.evaluate(test_data)
                print(f"Epoch {j + 1}: {eval} / {n_test} with accuracy {round(float(eval) / n_test * 100, 2)}%")
            else:
                print(f"Epoch {j + 1} complete")

            # Save current state of neural network
            np.save('biases.npy', self.biases)
            np.save('weights.npy', self.weights)


    def update_mini_batch(self, mini_batch, eta):
        """
        Update the network's weights and biases by applying gradient descent using backpropagation to a single mini batch
        :param mini_batch: list
            a small subset of the entire training data available to the network
        :param eta: float
            float representing the learning rate of the network, the bigger it is the bigger the adjustments will be
            made to the weights and biases every time the cost gradient is evaluated.
        """
        # Uses gradient descent and backprogation to adjust the weights and biases
        # according to the learning rate and error cost in the minibatch

        nabla_b = [np.zeros(b.shape) for b in self.biases]  # initialise gradients filled with 0s
        nabla_w = [np.zeros(w.shape) for w in self.weights]  # initialise gradients filled with 0s
        for x, y in mini_batch:
            # Determine what changes need to be made to the gradients
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)

            # Apply the changes according to the SGD update rule
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        # Apply the respective gradient descent update equations on biases and weights
        self.weights = [w - (eta / len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """
        Return a tuple (nabla_b, nabla_w) representing the gradient for the cost function C_x. They are in the same format
        as the biases and weights
        :param x: list
        :param y: list
        :return: nabla_b, nabla_w
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]  # initialise gradients
        nabla_w = [np.zeros(w.shape) for w in self.weights]  # initialise gradients

        # List to store all the activations, layer by layer. First activation layer is the image fed into the input layer.
        activations = [x]
        activation = x

        # List to store all the z vectors (wx + b), layer by layer
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # Final layer
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            # Start from second last layer and go backwards all the way to the first layer
            z = zs[-l]
            sp = sigmoid_prime(z)

            # Finds the partial derivative of the cost function WRT every bias and weight in the layer.
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())

        return nabla_b, nabla_w

    def evaluate(self, test_data):
        """
        Returns an integer representing how many cases the network guessed correctly
        :param test_data: list
            List containing the test data available to the network
        :return: amount_correct
        """
        # Takes the brightest neuron in the output layer after feeding the input to the neural network.
        # Puts them in tuples (a, y), where a was the output and y is the correct answer.
        test_results = [(np.argmax(self.feedfoward(x)), y) for (x, y) in test_data]

        # Returns how many of these tests were correctly identified.
        amount_correct = sum([int(x == y) for (x, y) in test_results])
        return amount_correct

    def cost_derivative(self, output_activations, y):
        # Returns the vector of partial derivatives dC_x/da for the output activations.
        # sum (a-y)^2
        return output_activations - y


# 50,000 training, ~ validation, 10,000 test
full_training_data, validation_data, full_test_data = load_data.load_data_wrapper()

# Splice training_data and test_data if necessary
training_data = full_training_data[:50000]
test_data = full_test_data[:1000]


def train_network():
    """
    Initialise a network and the train it using stochastic gradient descent
    """
    net = Network([784, 50, 20, 10])
    net.SGD(training_data, 40, 5, 0.025, test_data=test_data)

# train_network()

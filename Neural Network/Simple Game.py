import numpy as np


def sigmoid(x):
    return 1/(1+np.exp(-x))


def d_sigmoid(x):
    return x*(1-x)


inputs = np.array([[0, 0, 1],
                   [1, 1, 1],
                   [1, 0, 1],
                   [0, 1, 1]])

outputs = np.array([[0, 1, 1, 0]]).T

np.random.seed(1)

weights = 2*np.random.random((3, 1))-1

for iteration in range(20000):
    input_layer = inputs
    output_layer = sigmoid(np.dot(input_layer, weights))

    error = outputs - output_layer

    adjustments = error * d_sigmoid(output_layer)

    weights += np.dot(input_layer.T, adjustments)


print "Weights after training: "
print weights
print
print "Outputs after training: "
print output_layer
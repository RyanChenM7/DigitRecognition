'''
Imports the MNIST Dataset then splits the 60,000 cases
into a 50,000 learning set and 10,000 validation set.
'''

import numpy as np
import pickle
import gzip


def load_data():
    """
    Return the MNIST data as a tuple containing the training data,
    the validation data, and the test data.

    training_data is a ndarray containing 50,000 entries. Each entry is a tuple in the form (x, y),
    where x is a numpy ndarray with 784 values meant to represent the 28 x 28 = 784 pixels in a single MNIST image.
    y is a numpy ndarray containing the values from 0 to 9 corresponding to the value of the MNIST image.

    validation_data and test_data have the same format as training_data except only with 10,000 entries each.
    :return: (training_data, validation_data, test_data)
    """

    # Loads the MNIST database zip file. Change the path of the file accordingly.
    f = gzip.open('C:/mnist.pkl.gz', 'rb')
    format_pickle = pickle._Unpickler(f)
    format_pickle.encoding = 'latin1'
    training_data, validation_data, test_data = format_pickle.load()
    f.close()

    return training_data, validation_data, test_data


def load_data_wrapper():
    """
    Return a tuple containing training_data, validation_data and test_data
    Restructures data within training_data to be easier to use in the neural network.
    training_data is still a 50,000 entry ndarray with each entry as a tuple of the form (x, y). x is a 784-dimensional
    numpy ndarray representing the MNIST image. y is now a 10-dimensional numpy ndarray representing the unit vector
    corresponding to the correct digit for x.
    test_data and validation_data are the same format as training_data except y is still a single integer representing x
    :return: (training_data, validation_data, test_data)
    """

    # Load data then format them into the correct sizes of numpy arrays
    tr_d, va_d, te_d = load_data()
    # reshape numpy arrays
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = list(zip(training_inputs, training_results))
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = list(zip(validation_inputs, va_d[1]))
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = list(zip(test_inputs, te_d[1]))

    return training_data, validation_data, test_data


def vectorized_result(x):
    """
    Returns a 10D array filled with 0s with the value at index x equal to 1
    example: x = 2, y would be [0 ,0 ,2 ,0 ,0, 0, 0, 0, 0, 0]
    :param x: int
        integer to turn into a 10D vector
    :return: y
    """
    y = np.zeros((10, 1))
    # All are zeroes except for y[x]
    y[x] = 1
    return y


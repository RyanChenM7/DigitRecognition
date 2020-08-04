import neural_network as nnet
import numpy as np
import matplotlib.animation as mpa
import matplotlib.pyplot as plt


net = nnet.Network([784, 50, 20, 10], True)  # keep the already trained network


def get_activations():
    """Returns the results outputted by the network as a list of tuples in the form (x,y)
    x is an array of length 10, each index is a float, which is how confident the network is that the answer is that index
    y is the actual answer as an integer
    """
    return [(net.feedfoward(x), y) for (x, y) in nnet.test_data]


def show_results(picture_data, result, itr):
    """
    Creates two subplots, one is the actual handwritten digit as an image, the second is a bar graph indexed from 0 to 10,
    representing the certainty of the network in classifying each image.
    :param picture_data: list
        list in the form (x,y), x is an ndarray with shape (28,28) representing the handwritten digit in grayscale values,
        y is the value of the handwritten digit represented as an integer
    :param result: list
        list of tuples in the form (x,y), x is the output from the network, an array of length 10. y is the actual number
        as an integer
    :param itr: int
        An iterable
    """
    index = (itr + np.random.randint(0, 10000)) % len(picture_data)
    # show the picture of the number
    pic_data = np.reshape(picture_data[index][0], (28, 28))
    plt.subplot(1, 2, 1)
    plt.yticks([])
    plt.xticks([])
    plt.imshow(pic_data)
    # create the bar graph to represent network certainty
    plt.subplot(1, 2, 2)
    result_vector = [x[0] for x in result[index][0]]
    prediction = np.argmax(result_vector)  # predicted number, happens to be index
    actual = result[index][1]  # actual number
    caption = f"Prediction: {str(prediction)}    Answer: {str(actual)}"
    accuracy = f"Certainty: {str(round(100*np.max(result_vector), 1))}%"
    plt.xlabel(accuracy)
    plt.title(caption)
    # how certain the network is
    plt.ylim([0, 1])
    plt.xticks([i for i in range(11)])
    certainty_plot = plt.bar(range(10), result_vector, color="grey")
    if prediction == actual:  # change colour depending on if network guess was correct
        certainty_plot[prediction].set_color("green")
    else:
        certainty_plot[prediction].set_color("red")


def update_results(itr):
    """
    Function used to help create a matplotlib animation.
    Meant to cycle through the results given by the neural network when run on the test data
    :param itr: int
        An iterable
    """
    plt.clf()
    show_results(nnet.test_data, get_activations(), itr)


def run_visuals():
    """
    Creates the matplotlib figure and runs the animation
    """
    fig, ax = plt.subplots(figsize=(14, 7))
    ani_mst = mpa.FuncAnimation(fig, update_results, interval=2000)
    plt.show()


run_visuals()


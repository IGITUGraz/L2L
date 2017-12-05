import numpy as np


def sigmoid(x):
    """Compute sigmoid function for each element in x."""
    return 1.0 / (1.0 + np.exp(-x))


def relu(x):
    """ Compute RELU """
    result = x.copy()
    result[result < 0] = 0.
    return result


def softmax(a):
    """
    Compute softmax over the last dimension
    :param a:
    :return:
    """
    exp_a = np.exp(a - np.max(a))
    return exp_a / np.sum(exp_a, axis=-1)


class NeuralNetworkClassifier:
    def __init__(self, n_input, n_hidden, n_output):
        """

        :param n_input:
        :param n_hidden:
        :param n_output:
        """
        self.n_input, self.n_hidden, self.n_output = n_input, n_hidden, n_output
        self.hidden_weights = np.zeros((self.n_hidden, self.n_input))
        self.output_weights = np.zeros((self.n_output, self.n_hidden))

    def get_weights_shapes(self):
        """
        :return: A list of 2 tuples for each layer of the network
        """
        return [(self.n_hidden, self.n_input), (self.n_output, self.n_hidden)]

    def set_weights(self, hidden_weights, output_weights):
        self.hidden_weights[:] = hidden_weights
        self.output_weights[:] = output_weights

    def score(self, x, y):
        """

        :param x: batch_size x n_input size
        :param y: batch_size size
        :return:
        """
        hidden_activation = sigmoid(np.dot(self.hidden_weights, x.T))  # -> n_hidden x batch_size
        output_activation = np.dot(self.output_weights, hidden_activation)  # -> n_output x batch_size
        output_labels = np.argmax(output_activation, axis=0)  # -> batch_size
        assert y.shape == output_labels.shape, "The shapes of y and output labels are %s, %s" % (y.shape, output_labels.shape)
        n_correct = np.count_nonzero(y == output_labels)
        n_total = len(y)
        score = n_correct / n_total
        return score


def main():
    from sklearn.datasets import load_digits, fetch_mldata

    SMALL_MNIST = False

    if SMALL_MNIST:
        mnist_digits = load_digits()
        n_input = np.prod(mnist_digits.images.shape[1:])
        n_images = len(mnist_digits.images)  # 1797
        data_images = mnist_digits.images.reshape(n_images, -1) / 16.  # -> 1797 x 64
        data_targets = mnist_digits.target
        # im_size_x, im_size_y = 8, 8
    else:
        mnist_digits = fetch_mldata('MNIST original')
        n_input = np.prod(mnist_digits.data.shape[1:])
        data_images = mnist_digits.data / 255.  # -> 70000 x 284
        data_targets = mnist_digits.target
        # im_size_x, im_size_y = 28, 28

    n_hidden, n_output = 5, 10
    nn = NeuralNetworkClassifier(n_input, n_hidden, n_output)
    weight_shapes = nn.get_weights_shapes()
    weights = []
    for weight_shape in weight_shapes:
        weights.append(np.random.randn(*weight_shape))
    nn.set_weights(*weights)
    score = nn.score(data_images, data_targets)
    print("Score is: ", score)


if __name__ == '__main__':
    main()

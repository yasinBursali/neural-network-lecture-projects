import numpy as np
from sklearn.metrics import mean_squared_error
import math
# Input dataset for Training
X = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1]])

# Output class labels for Training
y = np.array([[0],
              [1],
              [1],
              [0]])

X1_test = np.array([1, 1, 0])
X2_test = np.array([1, 1, 1])


# eq 1 in the slide
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# eq 2 in the slide
def sigmoid_prime(z):
    # Note that z input into this function comes as a sigmoid value
    # therefore no need another sigmoid function for the derivative
    return (z * (1 - z))


def test_NN(X, w0, w1):
    z1 = np.dot(X, w0)
    a1 = sigmoid(z1)
    z2 = np.dot(a1, w1)
    a2 = sigmoid(z2)
    return a2


def SGD(X, y, alpha, epoch, batch_size):

    m, b = 0.33, 0.48  # initial parameters
    log, mse = [], []  # lists to store learning process

    for _ in range(epoch):
        indexes = np.random.randint(0, len(X), batch_size)  # random sample

        Xs = np.take(X, indexes)
        ys = np.take(y, indexes)
        N = len(Xs)

        f = ys - (m * Xs + b)

        # Updating parameters m and b
        m -= alpha * (-2 * Xs.dot(f).sum() / N)
        b -= alpha * (-2 * f.sum() / N)

        log.append((m, b))
        mse.append(math.sqrt(mean_squared_error(y, m * X + b)))


    return m, b, log, mse


# to ensure that generated random numbers
# are the same no matter how many times you run this
alpha = 0.4
batch = 10000

a0 = X
# weights, initialize it randomly
# make sure first weight matrix (weights connecting the input layer
# into hidden layer is 3 by 4
# and second weight matrix, weights connecting Hidden layer to output layer
# We assign random weights with values in the range -1 to 1
# and mean 0.
w0 = 2 * np.random.random((3, 4)) - 1  # Weights between input and first hidden layer
w1 = 2 * np.random.random((4, 1)) - 1

SGD(X,y,alpha,5,2)

print("\n********** QUESTION 2 **********\n")
print("Output for X1_test:")
print(test_NN(X1_test, w0, w1))
print("Output for X2_test:")
print(test_NN(X2_test, w0, w1))
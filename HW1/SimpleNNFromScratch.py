import numpy as np

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


alpha = 0.4
batch = 10000
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
        mse.append(mean_squared_error(y, m * X + b))

    return m, b, log, mse


# to ensure that generated random numbers
# are the same no matter how many times you run this


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
""""
for cntr in range(batch):

    batch_x = X
    batch_y = y
    n = batch_x.shape[0]

    # first input layer no activation functions
    a0 = batch_x

    # Perform Feedforward operation
    z1 = np.dot(a0, w0)
    a1 = sigmoid(z1)  # a1: activation values of the first layer, zeroth layer is input layer!

    # do the calcuation for the z values, notice that instead of sum operator
    # as we have seen in the class, we utilize the matrix operation dot product
    z2 = np.dot(a1, w1)

    a2 = sigmoid(z2)  # a2: second layer activation aka NN's output values

    l2_error = (a2 - batch_y) / n

    # print the total error sum for our gradient descent algorithm
    if cntr % 1000 == 0:
        print('Error:' + str(np.mean(np.mean(np.abs(l2_error)))))

    # eq. 6 in the slide
    l2_delta = l2_error * sigmoid_prime(a2)
    l1_error = l2_delta.dot(w1.T)

    # eq 7 in the slide
    l1_delta = l1_error * sigmoid_prime(a1)

    # eq 5 in the slide
    w1 -= alpha * a1.T.dot(l2_delta)
    w0 -= alpha * a0.T.dot(l1_delta)
"""
print('Output after training:')
print(a2)

print("\n********** QUESTION 2 **********\n")
print("Output for X1_test:")
print(test_NN(X1_test, w0, w1))
print("Output for X2_test:")
print(test_NN(X2_test, w0, w1))
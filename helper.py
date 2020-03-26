import numpy as np
import matplotlib.pyplot as plt
import h5py


def get_training_testing_data(train, test):

    train_file = h5py.File(train, 'r')
    test_file = h5py.File(test, 'r')

    x_train = train_file['train_set_x'].value
    y_train = train_file['train_set_y'].value
    x_test = test_file['test_set_x'].value
    y_test = test_file['test_set_y'].value

    train_file.close()
    test_file.close()

    # reshaping to convert X an Y to 2D array
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2] * x_train.shape[3])
    y_train = y_train.reshape(y_train.shape[0], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2] * x_test.shape[3])
    y_test = y_test.reshape(y_test.shape[0], 1)

    # normalising the values to floats between 0 to 1
    x_train = x_train / 255
    x_test = x_test / 255

    x_train = x_train.reshape(x_train.shape[0], -1).T
    x_test = x_test.reshape(x_test.shape[0], -1).T
    y_train = y_train.T
    y_test = y_test.T

    return x_train, y_train, x_test, y_test


def sigmoid(Z):

   A = 1/(1+np.exp(-Z))
   cache = Z

   return A, cache


def relu(Z):

   A = np.maximum(0,Z)
   assert(A.shape == Z.shape)
   cache = Z

   return A, cache


def relu_backward(dA, cache):

   Z = cache
   dZ = np.array(dA, copy=True)
   dZ[Z <= 0] = 0

   return dZ


def sigmoid_backward(dA, cache):

   Z = cache
   s = 1/(1+np.exp(-Z))
   dZ = dA * s * (1-s)

   return dZ


# CREATING THE MODEL

def L_layer_model(X, Y, layers_dims, learning_rate=0.005, num_iterations=3000, print_cost=False):
    """
    creates a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.

    X - data, np array of shape (num_px * num_px * 3, number of examples)
    Y -true label vector of shape (1, number of examples)
    layers_dims -list containing the input size and each layer size, of length (number of layers + 1).
    learning_rate -learning rate of the gradient descent update rule
    num_iterations - number of iterations of the optimization loop
    print_cost - print the cost every 100 steps if true

    Returns:
    parameters - parameters learnt by the model
    """

    np.random.seed(1)
    costs = []

    parameters = initialize_parameters_deep(layers_dims)

    for i in range(0, num_iterations):

        # forward propagation [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        AL, caches = L_model_forward(X, parameters)

        # compute cost
        cost = compute_cost(AL, Y)

        # backward propagation
        grads = L_model_backward(AL, Y, caches)

        # updates parameters
        parameters = update_parameters(parameters, grads, learning_rate)

        # print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)

    # plotting the cost
    plt.figure()
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('Number of iterations (per hundreds)')
    plt.title("Cross-entropy cost function evolution with learning rate =" + str(learning_rate))

    return parameters


def predict(parameters, X):

    """
    parameters - parameters of the model
    X - data, np array of shape (input size, nb of samples)

    Returns - the predicted data
    """

    m = X.shape[1]  # number of layers in the neural network
    p = np.zeros((1, m))

    # Forward propagation
    probas, caches = L_model_forward(X, parameters)  # convert probas to 0/1 predictions
    for i in range(0, probas.shape[1]):
        if probas[0, i] > 0.5:
            p[0, i] = 1
        else:
            p[0, i] = 0

    return p


def initialize_parameters(n_x, n_h, n_y):

    """
    n_x -size of the input layer
    n_h - size of the hidden layer
    n_y -size of the output layer

    Returns:
    parameters - python dictionaru containing parameters:
                    W1 - weight matrix of shape (n_h, n_x)
                    b1 -bias vector of shape (n_h, 1 )
                    W2 - weight matrix of shape (n_y,n_h)
                    b2 - bias vector of shape (n_y, 1)
    """

    np.random.seed(1)

    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters


def initialize_parameters_deep(layer_dims):
    """
    layer_dims - array containing the dimension of each layers in the network
    Returns:
    parameters - python dictionary containing:
                    Wl- weight matrix of shape ( layer_dims[l], layer_dims[l-1])
                    bl - bias vector of shape (layer_dims[l],1)
    """

    np.random.seed(1)
    parameters = {}
    L = len(layer_dims)  # number of layers in the network

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1])*np.sqrt(2./layer_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

    return parameters


def linear_forward(A, W, b):
    """
    A - activations from previous layer (or input data): (size of previous layer, number of examples)
    W -weights matrix: numpy array of shape : (size of current layer, size of previous layer)
    b - bias vector, np array of shape: (size of current layer, 1)

    Returns:
    Z - the inputof the activation function (pre-activation parameter)
    cache - tuple containing A, W and b
    """

    Z = W @ A + b
    cache = (A, W, b)

    return Z, cache


def linear_activation_forward(A_prev, W, b, activation):
    """
    creates the forward propagation for the LINEAR->ACTIVATION layer

    A_prev- activations from previous layer : (size of previous layer, number of examples)
    W - weights matrix: np array of shape (size of current layer, size of previous layer)
    b - bias vector, np array of shape (size of current layer, 1)
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    A - output of the activation function(post activation value)
    cache -tuple containing linear_cache and activation_cache
    """

    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)

    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)

    cache = (linear_cache, activation_cache)

    return A, cache


def L_model_forward(X, parameters):
    """
    creates forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation

    X - data, np array of shape (input size, nb of samples)
    parameters - the output of initialize_parameters_deep()

    Returns:
    AL - last post-activation value
    caches - list of caches containing:
                every cache of linear_activation_forward() (L-1 of them, indexed from 0 to L-1)
    """

    caches = []
    A = X
    L = len(parameters) // 2  # number of layers in the neural network

    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters["W" + str(l)], parameters["b" + str(l)],
                                             activation='relu')
        caches.append(cache)
    AL, cache = linear_activation_forward(A, parameters["W" + str(L)], parameters["b" + str(L)], activation='sigmoid')
    caches.append(cache)

    return AL, caches


def compute_cost(AL, Y):
    """
    AL- probability vector corresponding to the label predictions shape (1, number of examples)
    Y - true label vector shape (1, number of examples)

    Returns:
    cost - cross-entropy cost
    """

    m = Y.shape[1]
    cost = -np.sum((Y * np.log(AL) + (1 - Y) * np.log(1 - AL))) / m

    cost = np.squeeze(cost)

    return cost


def linear_backward(dZ, cache):
    """
    creates the linear portion of backward propagation for a single layer (layer l)

    dZ - gradient of the cost with respect to the linear output of current layer l
    cache - tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

    Returns:
    dA_prev - gradient of the cost with respect to the activation of the previous layer l-1 same shape as A_prev
    dW -gradient of the cost with respect to W current layer l, same shape as W
    db - gradient of the cost with respect to b current layer l, same shape as b
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = dZ @ A_prev.T / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    dA_prev = W.T @ dZ

    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation):
    """
    creates the backward propagation for the LINEAR->ACTIVATION layer.

    dA - post-activation gradient for current layer l
    cache - tuple of values (linear_cache, activation_cache)
    activation - the activation to be used, string sigmoid or relu

    Returns:
    dA_prev- gradient of the cost with respect to the activation of the previous layer l-1 same shape as A_prev
    dW - gradient of the cost with respect to W current layer l, same shape as W
    db -gradient of the cost with respect to b current layer l, same shape as b
    """
    linear_cache, activation_cache = cache

    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db


def L_model_backward(AL, Y, caches):
    """
    creates the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group

    AL- probability vector (output of the forward propagation (L_model_forward()) )
    Y- true label vector
    caches - list of caches that contains:
                every cache of linear_activation_forward() with relu
                the cache of linear_activation_forward() with sigmoid

    Returns:
    grads - dictionary with the gradients
    """
    grads = {}
    L = len(caches)  # number of layers
    Y = Y.reshape(AL.shape)

    # initializing backpropagation
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    current_cache = caches[L - 1]
    grads["dA" + str(L - 1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL,
                                                                                                      current_cache,
                                                                                                      activation='sigmoid')

    for l in reversed(range(L - 1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 1)], current_cache,
                                                                    activation='relu')
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads


def update_parameters(parameters, grads, learning_rate):
    """
    parameters- dict containing the parameters
    grads - dict containing the gradients (output of L_model_backward)

    Returns:
    parameters -dict containing the updated parameters
    """

    L = len(parameters) // 2  # number of layers in the neural network

    for l in range(L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]
    return parameters
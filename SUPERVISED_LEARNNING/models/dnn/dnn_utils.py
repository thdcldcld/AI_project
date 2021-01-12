import numpy as np
#dims_of_layers = [100, 10, 10, 5]
def init_params(dims_of_layers):
    # w 초기화: 가우시안, xavier initialization, He initialization...
    # b 초기화 0
    num_of_layers = len(dims_of_layers)
    params = {}
    for i in range(1, num_of_layers):
        params["w" + str(i)] = np.random.randn(dims_of_layers[i], dims_of_layers[i - 1]) * 0.01
        params["b" + str(i)] = np.zeros((dims_of_layers[i], 1))
    return params

def linear(w, b, a):
    z = np.matmul(w, a) + b
    linear_cache = w, b, a
    return z, linear_cache

def relu(z):
    a = np.maximum(0, z)
    activation_cache = z
    return a, activation_cache

def leaky_relu(z):
    a = np.maximum(0.01 * z, z)
    activation_cache = z
    return a, activation_cache

def sigmoid(z):
    a = 1 / (1 + np.exp(-z))
    activation_cache = z
    return a, activation_cache

def softmax(z):
    activation_cache = z
    a = np.exp(z) / np.sum(np.exp(z), axis=0, keepdims=True)
    return a, activation_cache

def single_forward(w, b, a, activation):
    z, linear_cache = linear(w, b, a)
    if activation == "relu":
        a, activation_cache = relu(z)
    elif activation == "leaky_relu":
        a, activation_cache = relu(z)
    elif activation == "sigmoid":
        a, activation_cache = sigmoid(z)
    elif activation == "softmax":
        a, activation_cache = softmax(z)

    linear_activation_cache = linear_cache, activation_cache
    return a, linear_activation_cache

def forward(params, x, activation, last_activation, num_of_layers):
    a = x
    forward_cache = []
    for i in range(1, num_of_layers):
        if i ! = num_of_layers - 1:
            a, linear_activation_cache = single_forward(params['w' + str(i)], params['b' + str(i)], a, activation)
        elif:
            a, linear_activation_cache = single_forward(params['w' + str(i)], params['b' + str(i)], a, last_activation)
        forward_cache.append(linear_activation_cache)

    return a, forward_cache

def cross_entropy(a, y):
    # 베르누이 확률분포
    m = y.shape[1]
    cost = np.sum(-(y * np.log(a))) / m
    return cost


def mean_square_error(a, y):
    # 가우시안 확률분포
    cost = np.sum(np.square(a - y)) / (2 * m)
    return cost

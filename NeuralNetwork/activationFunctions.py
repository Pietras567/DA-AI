import numpy as np

# Sigmoidalna funkcja aktywacji
def sigmoid(x):
    x = np.clip(x, -500, 500)  # Klampowanie wartości x
    return 1 / (1 + np.exp(-x))

# Pochodna funkcji sigmoidalnej
def sigmoid_derivative(x):
    sx = sigmoid(x)
    return sx * (1 - sx)

# ReLU funkcja aktywacji
def relu(x):
    return np.maximum(0, x)

# Pochodna funkcji ReLU
def relu_derivative(x):
    return np.where(x > 0, 1.0, 0.0)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x)**2

def leaky_relu(x):
    return np.where(x > 0, x, 0.01 * x)

def leaky_relu_derivative(x):
    return np.where(x > 0, 1.0, 0.01)

def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

def elu_derivative(x, alpha=1.0):
    return np.where(x > 0, 1.0, alpha * np.exp(x))

def softmax(x):
    e_x = np.exp(x - np.max(x))  # stabilizacja numeryczna
    return e_x / e_x.sum(axis=0)

def softmax_derivative(x):
    # Obliczamy softmax dla x
    s = softmax(x)
    # Tworzymy macierz Jacobiego
    return np.diagflat(s) - np.dot(s, s.T)


# Softplus funkcja aktywacji
def softplus(x):
    return np.log(1 + np.exp(x))

# Pochodna funkcji Softplus
def softplus_derivative(x):
    return 1 / (1 + np.exp(-x))

# Swish funkcja aktywacji
def swish(x, beta=1.0):
    return x * sigmoid(beta * x)

# Pochodna funkcji Swish
def swish_derivative(x, beta=1.0):
    sw = swish(x, beta)
    return beta * sw + sigmoid(beta * x) * (1 - beta * sw)

# Funkcja aktywacji liniowa
def linear(x):
    return x

# Pochodna funkcji liniowej
def linear_derivative(x):
    return np.ones_like(x)

def softsign(x):
    return x / (1 + np.abs(x))

def softsign_derivative(x):
    return 1 / (1 + np.abs(x))**2

# GELU funkcja aktywacji
def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

# Pochodna funkcji GELU
def gelu_derivative(x):
    cdf = 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
    pdf = np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)
    return cdf + x * pdf * 0.0356774 * (x**2 + 1)
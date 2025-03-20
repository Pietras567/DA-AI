import numpy as np

class Neuron:
    def __init__(self, num_inputs, num_outputs, activation_function, activation_derivative):
        self.bias = np.float64(0.05)  # Bias initialization
        self.weights = np.random.randn(num_inputs) * np.sqrt(2 / num_inputs)  # He initialization
        #self.weights = np.random.randn(num_inputs) * np.sqrt(2 / (num_inputs + num_outputs))  # Xavier (Glorot) initialization
        self.activation_function = activation_function
        self.activation_derivative = activation_derivative
        self.output = None
        self.delta = None

        # Adam optimizer parameters
        self.m_w, self.v_w = np.zeros_like(self.weights), np.zeros_like(self.weights)
        self.m_b, self.v_b = np.zeros_like(self.bias), np.zeros_like(self.bias)
        self.t = 0  # time step

    def feedforward(self, inputs):

        self.inputs = inputs
        #print(self.inputs)
        #print(self.weights)
        #print(self.bias)
        value = np.dot(self.inputs, self.weights)
        value = value + self.bias
        #print(value)
        self.output = self.activation_function(value)
        return self.output

    def calculate_delta(self, target=None, forward_weights=None, forward_deltas=None):
        if target is not None:
            # Output neuron
            self.delta = (target - self.output) * self.activation_derivative(self.output)
        else:
            # Hidden neuron
            self.delta = np.dot(forward_weights, forward_deltas) * self.activation_derivative(self.output)

        if np.isnan(self.delta).any() or np.isinf(self.delta).any():
            print(
                f"Warning: delta contains NaN values. Inputs: {self.inputs}, Weights: {self.weights}, Output: {self.output}")
    def update_weights(self, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8, lambda_l1=0.00, lambda_l2=0.00):
        self.t += 1

        lr_t = learning_rate * np.sqrt(1 - beta2 ** self.t) / (1 - beta1 ** self.t)

        # Update weights
        grad_w = self.delta * self.inputs
        # Apply L1 and L2 regularization
        grad_w += lambda_l1 * np.sign(self.weights) + lambda_l2 * self.weights

        self.m_w = beta1 * self.m_w + (1 - beta1) * grad_w
        self.v_w = beta2 * self.v_w + (1 - beta2) * (grad_w ** 2)
        m_w_hat = self.m_w / (1 - beta1 ** self.t)
        v_w_hat = self.v_w / (1 - beta2 ** self.t)
        self.weights += lr_t * m_w_hat / (np.sqrt(v_w_hat) + epsilon) - learning_rate * lambda_l2 * self.weights

        # Update bias
        grad_b = self.delta
        self.m_b = beta1 * self.m_b + (1 - beta1) * grad_b
        self.v_b = beta2 * self.v_b + (1 - beta2) * (grad_b ** 2)
        m_b_hat = self.m_b / (1 - beta1 ** self.t)
        v_b_hat = self.v_b / (1 - beta2 ** self.t)
        self.bias += lr_t * m_b_hat / (np.sqrt(v_b_hat) + epsilon)

        if np.isnan(self.weights).any():
            print(
                f"Warning: delta contains NaN or inf values. Inputs: {self.inputs}, Weights: {self.weights}, Output: {self.output}, Delta: {self.delta}")
        #print(f"Updated weights: {self.weights}")
        #print(f'Bias: {self.bias}')


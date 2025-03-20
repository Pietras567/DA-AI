import numpy as np
from neuron import *

class Layer:
    def __init__(self, num_neurons, num_inputs, num_outputs, activation_functions, activation_derivatives, dropout_rate=0.0):
        #self.neurons = [Neuron(num_inputs_per_neuron, af, ad) for af, ad in zip(activation_functions, activation_derivatives)]
        self.neurons = [Neuron(num_inputs, num_outputs, activation_functions[i], activation_derivatives[i]) for i in range(num_neurons)]
        self.dropout_rate = dropout_rate
    def feedforward(self, inputs):
        self.outputs = np.array([neuron.feedforward(inputs) for neuron in self.neurons])
        #print(f"Layer outputs: {self.outputs}")

        # Apply dropout
        if self.dropout_rate > 0:
            self.dropout_mask = np.random.binomial(1, 1 - self.dropout_rate, size=self.outputs.shape) / (
                        1 - self.dropout_rate)
            self.outputs *= self.dropout_mask

        return self.outputs

    def calculate_deltas(self, targets=None, forward_layer=None):
        if targets is not None:
            for neuron, target in zip(self.neurons, targets):
                neuron.calculate_delta(target=target)
                #print(len(targets))
        else:
            forward_weights = np.array([neuron.weights[:] for neuron in forward_layer.neurons])
            #print(len(forward_weights))
            forward_deltas = np.array([neuron.delta for neuron in forward_layer.neurons])
            for i, neuron in enumerate(self.neurons):
                neuron.calculate_delta(forward_weights=forward_weights[:, i], forward_deltas=forward_deltas)
                #print(i)
                #print(len(forward_weights[:, i]))
        #print(f"Layer deltas: {[neuron.delta for neuron in self.neurons]}")

    def update_weights(self, learning_rate):
        for neuron in self.neurons:
            neuron.update_weights(learning_rate)


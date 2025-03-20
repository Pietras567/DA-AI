from layer import *
import numpy as np
import pickle
class NeuralNetwork:
    def __init__(self, layer_sizes, activation_functions, activation_derivatives, dropout_rates=None):
        self.layers = []
        if dropout_rates is None:
            dropout_rates = [0.0] * (len(layer_sizes) - 1)
        for i in range(len(layer_sizes) - 1):
            num_inputs = layer_sizes[i]
            if i != (len(layer_sizes) - 2):
                num_outputs = layer_sizes[i+2]
            else:
                num_outputs = 0
            num_neurons = layer_sizes[i + 1]
            af = activation_functions[i]
            ad = activation_derivatives[i]
            dr = dropout_rates[i]
            self.layers.append(Layer(num_neurons, num_inputs, num_outputs, af, ad, dr))
            #print(dr)
        #print(self.layers)
    def feedforward(self, inputs):
        for layer in self.layers:
            inputs = layer.feedforward(inputs)
        return inputs

    def train(self, training_data, epochs, learning_rate, batch_size=32, early_stopping=False, patience=10, validation_data=None):
        best_loss = float('inf')
        patience_counter = 0
        mse_history = []  # List to store MSE each epoch
        mae_history = []  # List to store MAE each epoch

        for epoch in range(epochs):
            np.random.shuffle(training_data)
            batches = [training_data[k:k + batch_size] for k in range(0, len(training_data), batch_size)]
            total_mse = 0
            total_mae = 0

            for batch in batches:
                batch_mse = 0
                batch_mae = 0
                for inputs, targets in batch:
                    # Forward pass
                    outputs = self.feedforward(inputs)

                    mse = np.mean((targets - outputs) ** 2)
                    mae = np.mean(np.abs(targets - outputs))
                    batch_mse += mse
                    batch_mae += mae

                    # Backward pass
                    self.layers[-1].calculate_deltas(targets=targets)
                    for i in range(len(self.layers) - 2, -1, -1):
                        self.layers[i].calculate_deltas(forward_layer=self.layers[i + 1])

                # Update weights after processing each batch
                for layer in self.layers:
                    layer.update_weights(learning_rate)

                total_mse += batch_mse
                total_mae += batch_mae

            mse_history.append(total_mse / len(training_data))
            mae_history.append(total_mae / len(training_data))
            print(f"Epoch {epoch + 1}/{epochs}, MSE: {total_mse / len(training_data)}, MAE: {total_mae / len(training_data)}")

            # Early stopping
            if validation_data:
                validation_error = self.evaluate(validation_data)
                if validation_error < best_loss:
                    best_loss = validation_error
                    patience_counter = 0
                else:
                    patience_counter += 1

                if early_stopping and patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

        return mse_history, mae_history



    def evaluate(self, data):
        total_error = 0
        for inputs, targets in data:
            outputs = self.feedforward(inputs)
            error = np.mean((targets - outputs) ** 2)
            total_error += error
        return total_error / len(data)

def save_model(model, file_name, scalersX, scalersY):
    modelName = file_name + '.pickle'
    NameScalersX = file_name + '_scalersX.pickle'
    NameScalersY = file_name + '_scalersY.pickle'

    with open(modelName, 'wb') as file:
        pickle.dump(model, file)
    with open(NameScalersX, 'wb') as file1:
        pickle.dump(scalersX, file1)
    with open(NameScalersY, 'wb') as file2:
        pickle.dump(scalersY, file2)

def load_model(file_name):
    modelName = file_name + '.pickle'
    NameScalersX = file_name + '_scalersX.pickle'
    NameScalersY = file_name + '_scalersY.pickle'

    with open(modelName, 'rb') as file:
        model = pickle.load(file)
    with open(NameScalersX, 'rb') as file1:
        scalersX = pickle.load(file1)
    with open(NameScalersY, 'rb') as file2:
        scalersY = pickle.load(file2)

    return model, scalersX, scalersY

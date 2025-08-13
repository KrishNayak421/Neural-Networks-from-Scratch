"""
Neural Network Model

This module combines all components to create a complete neural network
"""

import numpy as np
from layers import Layer_Dense
from activations import Activation_ReLU, Activation_Sigmoid, Activation_Softmax
from losses import Loss_CategoricalCrossentropy
from data_utils import create_spiral_data, plot_data

class NeuralNetwork:
    
    def __init__(self):
        self.layers = []
        self.activations = []
        self.loss_function = None
    
    def add_layer(self, layer):
        self.layers.append(layer)
    
    def add_activation(self, activation):
        self.activations.append(activation)
    
    def set_loss(self, loss_function):
        self.loss_function = loss_function
    
    def forward(self, X):
       
        current_input = X
        
        # Forward pass through each layer and activation
        for i, (layer, activation) in enumerate(zip(self.layers, self.activations)):
            # Layer forward pass
            layer.forward(current_input)
            # Activation forward pass
            activation.forward(layer.output)
            # Update input for next layer
            current_input = activation.output
        
        return current_input
    
    def calculate_loss(self, predictions, y_true):
        if self.loss_function is None:
            raise ValueError("Loss function not set. Use set_loss() method.")
        return self.loss_function.calculate(predictions, y_true)
    
    def predict(self, X):
        return self.forward(X)

def create_sample_network():

    # Set random seed for reproducibility
    np.random.seed(0)
    
    # Create network
    network = NeuralNetwork()
    
    # Add layers and activations
    # Input layer: 2 features -> 3 neurons
    network.add_layer(Layer_Dense(2, 3))
    network.add_activation(Activation_ReLU())
    
    # Output layer: 3 neurons -> 3 classes
    network.add_layer(Layer_Dense(3, 3))
    network.add_activation(Activation_Softmax())
    
    # Set loss function
    network.set_loss(Loss_CategoricalCrossentropy())
    
    return network

if __name__ == "__main__":
    print("Creating and testing neural network...")
    
    network = create_sample_network() #Creating sample network
    
    X, y = create_spiral_data(100, 3)
    print(f"Dataset shape: {X.shape}, Labels shape: {y.shape}")
    
    predictions = network.forward(X)
    print(f"Predictions shape: {predictions.shape}")
    print("First 5 predictions:\n", predictions[:5])
    
    loss = network.calculate_loss(predictions, y)
    print(f"\nInitial loss: {loss:.4f}")
    
    predicted_classes = np.argmax(predictions, axis=1)
    accuracy = np.mean(predicted_classes == y)
    print(f"Initial accuracy: {accuracy:.4f}")
    
    print("\nFirst 5 samples:")
    for i in range(5):
        print(f"Sample {i+1}: True class = {y[i]}, "
              f"Predicted probs = {predictions[i]}, "
              f"Predicted class = {predicted_classes[i]}")
    

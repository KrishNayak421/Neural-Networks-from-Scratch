"""
Dense Layer Implementation
"""

import numpy as np

class Layer_Dense:
    
    def __init__(self, n_inputs, n_neurons):
        
        # Initialize weights with small random values
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        # Initialize biases to zero
        self.biases = np.zeros((1, n_neurons))
    
    def forward(self, inputs):
        
        # Store inputs for potential backward pass
        self.inputs = inputs
        # Calculate output: inputs * weights + biases
        self.output = np.dot(inputs, self.weights) + self.biases

# Example usage and testing
if __name__ == "__main__":
    # Set seed for reproducibility
    np.random.seed(0)
    
    # Create test data
    test_inputs = np.array([[1, 2, 3, 4],
                           [5, 4, 5, 6],
                           [1, 7, 8, 9]])
    
    # Create layers
    layer1 = Layer_Dense(4, 3)  # 4 inputs, 3 neurons
    layer2 = Layer_Dense(3, 2)  # 3 inputs, 2 neurons
    
    # Forward pass through first layer
    layer1.forward(test_inputs)
    print("Layer 1 output shape:", layer1.output.shape)
    print("Layer 1 output:\n", layer1.output)
    
    # Forward pass through second layer
    layer2.forward(layer1.output)
    print("\nLayer 2 output shape:", layer2.output.shape)
    print("Layer 2 output:\n", layer2.output)

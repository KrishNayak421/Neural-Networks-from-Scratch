"""
Activation Functions Implementation
"""

import numpy as np

# Individual activation functions
def relu(x): 
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Activation classes
class Activation_ReLU:
    def __init__(self):
        pass
    
    def forward(self, inputs):
        # Store inputs for potential backward pass
        self.inputs = inputs
        # Apply ReLU activation
        self.output = relu(inputs)

class Activation_Sigmoid:
    def __init__(self):
        pass
    
    def forward(self, inputs):

        # Store inputs for potential backward pass
        self.inputs = inputs
        # Apply Sigmoid activation
        self.output = sigmoid(inputs)

class Activation_Softmax:
    def __init__(self):
        pass
    
    def forward(self, inputs):
        
        # Store inputs for potential backward pass
        self.inputs = inputs
        
        # Subtract max for numerical stability (prevents overflow)
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        
        # Calculate probabilities (normalize)
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        
        self.output = probabilities

# Example usage and testing
if __name__ == "__main__":
    # Test data
    test_inputs = np.array([[-1.0, 0.0, 1.0, 2.0],
                           [0.5, -0.5, 1.5, -2.0],
                           [0.1, 0.2, 0.3, 0.4]])
    
    print("Test inputs:\n", test_inputs)
    
    # Test ReLU
    relu_activation = Activation_ReLU()
    relu_activation.forward(test_inputs)
    print("\nReLU output:\n", relu_activation.output)
    
    # Test Sigmoid
    sigmoid_activation = Activation_Sigmoid()
    sigmoid_activation.forward(test_inputs)
    print("\nSigmoid output:\n", sigmoid_activation.output)
    
    # Test Softmax
    softmax_activation = Activation_Softmax()
    softmax_activation.forward(test_inputs)
    print("\nSoftmax output:\n", softmax_activation.output)
    print("Softmax sum per sample:", np.sum(softmax_activation.output, axis=1))

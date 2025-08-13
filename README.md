# Neural Networks from Scratch

A comprehensive implementation of neural networks built from scratch using NumPy, demonstrating fundamental concepts of deep learning without relying on high-level frameworks.

## References:

videos : 
- neural networks : https://youtu.be/Wo5dMEP_BbI?si=cgtnFPskOVnqrqtQ
- back propogation : https://youtu.be/SmZmBKc7Lrs?si=ru9ffLOAWk2rWZ6o 

texts:
- https://cs231n.github.io/optimization-2/#staged 
- https://pyimagesearch.com/2021/05/06/backpropagation-from-scratch-with-python/

## Overview

This project implements a neural network from the ground up, covering:
- Dense (fully connected) layers
- Activation functions (ReLU, Sigmoid, Softmax)
- Loss functions (Categorical Cross-entropy)
- Forward propagation
- Data generation and visualization

## Features

- **Custom Layer Implementation**: Built a `Layer_Dense` class for fully connected layers
- **Activation Functions**: 
  - ReLU (Rectified Linear Unit)
  - Sigmoid
  - Softmax (with numerical stability improvements)
- **Loss Functions**: Categorical Cross-entropy loss
- **Data Generation**: Spiral dataset generator for multi-class classification
- **Visualization**: Data plotting using matplotlib

## Project Structure

```
Neural Networks from Scratch/
├── code.ipynb              # Original notebook implementation
├── README.md               # Project documentation
├── layers.py               # Dense layer implementation
├── activations.py          # Activation functions (ReLU, Sigmoid, Softmax)
├── losses.py               # Loss functions (Categorical Cross-entropy)
├── data_utils.py           # Data generation and visualization utilities
├── model.py                # Neural network model combining all components
└── examples.py             # Usage examples and demonstrations
```

## Module Organization

The implementation has been organized into modular Python files for better maintainability and understanding:

### `layers.py` - Neural Network Layers
- **Layer_Dense**: Fully connected layer implementation
- Forward propagation through dense layers
- Weight initialization and bias handling
- Supports batch processing

### `activations.py` - Activation Functions
- **Activation_ReLU**: ReLU activation function (f(x) = max(0, x))
- **Activation_Sigmoid**: Sigmoid activation function (f(x) = 1/(1 + e^(-x)))
- **Activation_Softmax**: Softmax activation with numerical stability
- Individual helper functions: `relu()`, `sigmoid()`

### `losses.py` - Loss Functions
- **Loss**: Base loss class with common functionality
- **Loss_CategoricalCrossentropy**: Multi-class classification loss
- Handles both sparse labels and one-hot encoded labels
- Includes numerical stability with clipping

### `data_utils.py` - Data Generation & Utilities
- **create_spiral_data()**: Generate spiral dataset for multi-class testing
- **plot_data()**: Visualization utilities for 2D datasets
- **create_simple_data()**: Simple test data generators
- Matplotlib integration for data visualization

### `model.py` - Complete Neural Network
- **NeuralNetwork**: Main network class combining all components
- Manages layers, activations, and loss functions
- **create_sample_network()**: Pre-configured example network
- Forward propagation through entire network pipeline

### `examples.py` - Usage Demonstrations
- Comprehensive usage examples for all components
- Step-by-step tutorials and demonstrations
- **run_all_examples()**: Execute all demonstrations
- Testing and validation examples

## 🛠️ Implementation Details

### Dense Layer
```python
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1 * np.random.rand(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
```

### Activation Functions
- **ReLU**: `f(x) = max(0, x)`
- **Sigmoid**: `f(x) = 1 / (1 + e^(-x))`
- **Softmax**: Normalized exponential function for probability distribution

### Loss Function
- **Categorical Cross-entropy**: Measures the difference between predicted and actual probability distributions

## Dataset

The project uses a synthetic spiral dataset with:
- 3 classes
- 100 points per class
- 2D feature space
- Visualized using scatter plots with color-coded classes

## Requirements

```python
import numpy as np
import matplotlib.pyplot as plt
import math
```

## Getting Started

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/neural-networks-from-scratch.git
   cd neural-networks-from-scratch
   ```

2. **Install dependencies**:
   ```bash
   pip install numpy matplotlib
   ```

3. **Run the examples**:
   ```bash
   python examples.py          # Run all demonstrations
   ```

4. **Or explore individual modules**:
   ```bash
   python layers.py            # Test dense layers
   python activations.py       # Test activation functions
   python losses.py            # Test loss functions
   python data_utils.py        # Test data generation
   python model.py             # Test complete network
   ```

5. **Or use the original notebook**:
   Open `code.ipynb` in Jupyter Notebook or any compatible environment.

## Usage Example

### Quick Start with Modular Components:
```python
# Import the modules
from model import create_sample_network
from data_utils import create_spiral_data

# Create network and data
network = create_sample_network()
X, y = create_spiral_data(100, 3)

# Forward pass
predictions = network.forward(X)
loss = network.calculate_loss(predictions, y)

print(f"Loss: {loss:.4f}")
```

### Building a Custom Network:
```python
from layers import Layer_Dense
from activations import Activation_ReLU, Activation_Softmax
from losses import Loss_CategoricalCrossentropy

# Build network manually
layer1 = Layer_Dense(2, 5)      # Input layer
activation1 = Activation_ReLU()
layer2 = Layer_Dense(5, 3)      # Output layer  
activation2 = Activation_Softmax()
loss_fn = Loss_CategoricalCrossentropy()

# Forward pass
layer1.forward(X)
activation1.forward(layer1.output)
layer2.forward(activation1.output)
activation2.forward(layer2.output)

# Calculate loss
loss = loss_fn.calculate(activation2.output, y)
```

### Original Notebook Style:
```python
# Create synthetic data
X, y = create_data(100, 3)

# Build network
layer1 = Layer_Dense(2, 3)
activation1 = Activation_Relu()
layer2 = Layer_Dense(3, 3)
activation2 = Activation_Softmax()

# Forward pass
layer1.forward(X)
activation1.forward(layer1.output)
layer2.forward(activation1.output)
activation2.forward(layer2.output)

# Calculate loss
loss_function = Loss_CategoricalCrossentropy()
loss = loss_function.calculate(activation2.output, y)
print(f"Loss: {loss}")
```

## Learning Objectives

This project demonstrates:
- Understanding of neural network fundamentals
- Implementation of forward propagation
- Working with matrices and vectorization
- Numerical stability in computations
- Object-oriented programming for ML components
- **Modular code organization and software engineering practices**
- Building reusable components for machine learning

## Future Enhancements (Upcoming)

- [ ] Backward propagation implementation
- [ ] Gradient descent optimization
- [ ] Additional activation functions (Tanh, Leaky ReLU)
- [ ] Regularization techniques
- [ ] Model evaluation metrics
- [ ] More complex architectures
- [ ] Training loops with epochs and batches
- [ ] Model saving and loading capabilities

## Testing the Implementation

Each module includes built-in tests that you can run:

```bash
# Test all components with examples
python examples.py

# Test individual modules
python layers.py        # Tests Layer_Dense class
python activations.py   # Tests all activation functions  
python losses.py        # Tests loss calculation
python data_utils.py    # Tests data generation
python model.py         # Tests complete network
```


##  Future Enhancements (To be added later)

- [ ] Backward propagation implementation
- [ ] Gradient descent optimization
- [ ] Additional activation functions (Tanh, Leaky ReLU)
- [ ] Regularization techniques
- [ ] Model evaluation metrics
- [ ] More complex architectures


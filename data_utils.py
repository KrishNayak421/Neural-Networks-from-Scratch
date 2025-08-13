"""
Data Generation and Utilities
"""

import numpy as np
import matplotlib.pyplot as plt

def create_spiral_data(points, classes):
    """
    Create a spiral dataset for multi-class classification
    
    Args:
        points: Number of points per class
        classes: Number of classes
        
    Returns:
        X: Feature matrix (points*classes, 2)
        y: Labels (points*classes,)
    """
    # Initialize arrays
    X = np.zeros((points * classes, 2))
    y = np.zeros(points * classes, dtype='uint8')
    
    # Generate data for each class
    for class_number in range(classes):
        # Calculate indices for current class
        ix = range(points * class_number, points * (class_number + 1))
        
        # Generate radius values (0 to 1)
        r = np.linspace(0.0, 1, points)
        
        # Generate theta values with some noise
        t = np.linspace(class_number * 4, (class_number + 1) * 4, points) + \
            np.random.randn(points) * 0.2
        
        # Convert to Cartesian coordinates
        X[ix] = np.c_[r * np.sin(t * 2 * np.pi), r * np.cos(t * 2 * np.pi)]
        
        # Assign class labels
        y[ix] = class_number
    
    return X, y

def plot_data(X, y, title="Data Visualization", figsize=(10, 6)):
    
    plt.figure(figsize=figsize)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.colorbar()
    plt.show()

def create_simple_data():
    
    # Basic inputs for layer testing
    inputs_1d = np.array([1, 2, 3])
    inputs_2d = np.array([[1, 2, 3, 4],
                         [5, 4, 5, 6],
                         [1, 7, 8, 9]])
    inputs_batch = np.array([[3, 4, 1],
                            [1, 2, 3],
                            [2, 1, 4]])
    
    return inputs_1d, inputs_2d, inputs_batch

# Example usage and testing
if __name__ == "__main__":
    # Generate spiral data
    print("Generating spiral dataset...")
    X, y = create_spiral_data(100, 3)
    
    print(f"Dataset shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Number of classes: {len(np.unique(y))}")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Show first few samples
    print("\nFirst 5 samples:")
    print("Features:\n", X[:5])
    print("Labels:", y[:5])
    
    # Plot the data
    plot_data(X, y, "Spiral Dataset - 3 Classes")
    
    # Generate simple test data
    print("\nSimple test data:")
    inputs_1d, inputs_2d, inputs_batch = create_simple_data()
    print("1D inputs:", inputs_1d)
    print("2D inputs:\n", inputs_2d)
    print("Batch inputs:\n", inputs_batch)

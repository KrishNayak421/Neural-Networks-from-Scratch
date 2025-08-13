"""
Loss Functions Implementation
"""

import numpy as np

class Loss:
    
    def __init__(self):
        pass
    
    def calculate(self, output, y_true):
        
        # Calculate sample losses
        sample_losses = self.forward(output, y_true)
        # Calculate mean loss
        data_loss = np.mean(sample_losses)
        return data_loss

class Loss_CategoricalCrossentropy(Loss):
    
    def __init__(self):
        super().__init__()
    
    def forward(self, y_pred, y_true):
        
        samples = len(y_pred)
        
        # Clip predicted values to prevent division by zero
        # Values are clipped to range [1e-7, 1-1e-7]
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        
        # Handle different label formats
        if len(y_true.shape) == 1:
            # Sparse labels (e.g., [0, 1, 2])
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            # One-hot encoded labels (e.g., [[1,0,0], [0,1,0], [0,0,1]])
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
        
        # Calculate negative log-likelihood
        negative_log_likelihoods = -np.log(correct_confidences)
        
        return negative_log_likelihoods

# Example usage and testing
if __name__ == "__main__":
    # Test data - 3 samples, 3 classes
    # Softmax outputs (probabilities)
    predictions = np.array([[0.7, 0.2, 0.1],
                           [0.1, 0.8, 0.1],
                           [0.2, 0.3, 0.5]])
    
    # True labels (sparse format)
    true_labels_sparse = np.array([0, 1, 2])
    
    # True labels (one-hot format)
    true_labels_onehot = np.array([[1, 0, 0],
                                  [0, 1, 0],
                                  [0, 0, 1]])
    
    print("Predictions:\n", predictions)
    print("True labels (sparse):", true_labels_sparse)
    print("True labels (one-hot):\n", true_labels_onehot)
    
    # Test loss function
    loss_function = Loss_CategoricalCrossentropy()
    
    # Test with sparse labels
    loss_sparse = loss_function.calculate(predictions, true_labels_sparse)
    print(f"\nLoss (sparse labels): {loss_sparse:.4f}")
    
    # Test with one-hot labels
    loss_onehot = loss_function.calculate(predictions, true_labels_onehot)
    print(f"Loss (one-hot labels): {loss_onehot:.4f}")
    
    # Test individual sample losses
    sample_losses = loss_function.forward(predictions, true_labels_sparse)
    print(f"Individual sample losses: {sample_losses}")

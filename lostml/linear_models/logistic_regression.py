import numpy as np
from .base import _BaseRegression

class LogisticRegression(_BaseRegression):
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        super().__init__(learning_rate, n_iterations)
        self.weights = None
        self.bias = None

    def _get_gradients(self, X, y, y_pred):
        N = X.shape[0]
        # --- Gradient for weights ---
        # Formula: (1/N) * X^T • (y_pred - y)
        dw = (1/N) * np.dot(X.T, (y_pred-y))
        db = (1/N) * np.sum(y_pred-y)
        return dw, db
    
    def sigmoid(self, z):
        # Sigmoid function: f(z) = 1 / (1 + exp(-z))
        return 1 / (1 + np.exp(-z))
    
    def predict(self, X):
        linear_pred = super().predict(X)
        # Apply sigmoid function to get probabilities
        return self.sigmoid(linear_pred)
    
    def predict_proba(self, X):
        # Get probabilities for each class
        probabilities = self.predict(X)
        # Combine probabilities for each class
        return np.c_[1-probabilities, probabilities]
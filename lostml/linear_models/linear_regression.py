import numpy as np
from .base import _BaseRegression

class LinearRegression(_BaseRegression):
    def _get_gradients(self, X, y, y_pred):
        N = X.shape[0]
        # --- Gradient for weights ---
        # Formula: (1/N) * X^T • (y_pred - y)
        dw = (1/N) * np.dot(X.T, (y_pred-y))

        # --- Gradient for Bias ---
        db = (1/N) * np.sum(y_pred-y)
        return dw, db
    

class RigdeRegression(_BaseRegression):
    def __init__(self, lambda_param=1.0, learning_rate=0.01, n_iterations=1000):
        super.__init__(learning_rate, n_iterations)
        self.alpha = lambda_param

    def _get_gradients(self, X, y, y_pred):
        N = X.shape[0]
        # --- Gradient for weights ---
        # Formula: (1/N) * X^T • (y_pred - y)
        mse_dw = (1/N) * np.dot(X.T, (y_pred-y))
        mse_db = (1/N) * np.sum(y_pred-y)

        # Add penalty
        # Original cost: J(θ) = MSE + α * Σ(θ_j²)
        # Derivative of penalty w.r.t θ_j = 2 * α * θ_j
        # In vector form, this is 2 * α * self.weights
        dw = mse_dw + 2 * self.alpha * self.weights
        db = mse_db  # We don't penalise bias
        return dw, db

class LassoRegression(_BaseRegression):
    def __init__(self, lambda_param=1.0, learning_rate=0.01, n_iterations=1000):
        super.__init__(learning_rate, n_iterations)
        self.alpha = lambda_param
    
    def _get_gradients(self, X, y, y_pred):
        N = X.shape[0]
        # --- Gradient for weights ---
        # Formula: (1/N) * X^T • (y_pred - y)
        mse_dw = (1/N) * np.dot(X.T, (y_pred-y))
        mse_db = (1/N) * np.sum(y_pred-y)

        # Add penalty
        # Original cost: J(θ) = MSE + α * Σ|θ_j|
        # Derivative of penalty w.r.t θ_j = α * sign(θ_j)
        # We use np.sign() which returns +1, -1, or 0.
        dw = mse_dw + self.alpha * np.sign(self.weights)
        db = mse_db
        return dw, db
    
class ElasticNet(_BaseRegression):
    def __init__(self, alpha=1.0, l1_ratio=0.5, learning_rate=0.01, n_iterations=1000):
        super().__init__(learning_rate, n_iterations)
        self.alpha = alpha
        self.l1_ratio = l1_ratio

    def _get_gradients(self, X, y, y_pred):
        N = X.shape[0]
        # --- Gradient for weights ---
        # Formula: (1/N) * X^T • (y_pred - y)
        mse_dw = (1/N) * np.dot(X.T, (y_pred-y))
        mse_db = (1/N) * np.sum(y_pred-y)

        # Lambda
        l1_lambda = self.alpha * self.l1_ratio
        l2_lambda = self.alpha * (1-self.l1_ratio)

        # Penalty
        l1_penalty = l1_lambda * np.sign(self.weights)
        l2_penalty = l2_lambda * 2 * self.weights

        # Gradients
        dw = mse_dw + l1_penalty + l2_penalty
        db = mse_db
        return dw, db
    

    
    


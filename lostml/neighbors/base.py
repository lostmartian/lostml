import numpy as np

class _BaseNeighbors:
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None
    
    def fit(self, X, y):
        """
        Store training data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        # Validate inputs
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples")
        
        if self.n_neighbors > X.shape[0]:
            raise ValueError(
                f"n_neighbors ({self.n_neighbors}) cannot be greater than "
                f"number of samples ({X.shape[0]})"
            )
        
        self.X_train = X
        self.y_train = y
    
    def predict(self, y_test):
        """Predict using stored training data."""
        raise NotImplementedError()
    
    def _check_fitted(self):
        """Check if model has been fitted."""
        if self.X_train is None or self.y_train is None:
            raise ValueError("This instance has not been fitted yet. Call 'fit' first.")

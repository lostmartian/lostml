import numpy as np

class _BaseRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.n_iter = n_iterations
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        # Start with some initial guess
        self.weights = np.zeros(n_features)
        self.bias = 0

        # -- Gradient Descent Loop --- 
        for _ in range(self.n_iter):
            # Predict y
            y_pred = self.predict(X)

            # Calculate gradient based on the algorithm used
            dw, db = self._get_gradients(X, y, y_pred)

            # --- Update the weights and bias ---
            # We move a small step (self.lr) in the downhill direction (dw, db)
            self.weights -= self.lr*dw
            self.bias -= self.lr*db
    
    def predict(self, X):
        # h(x) = X*w + b
        return np.dot(X, self.weights) + self.bias
    
    def _get_gradients(self, X, y, y_pred):
        # This is an "abstract" method. It's meant to be
        # replaced by the "child" classes (like LinearRegression).
        raise NotImplementedError()

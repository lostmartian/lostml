import numpy as np
from .base import _BaseNeighbors
from ..utils.distances import euclidean_distance, manhattan_distance

class KNN(_BaseNeighbors):
    def __init__(self, n_neighbors=5, metric='euclidean'):
        super().__init__(n_neighbors)
        self.metric = metric

    def _get_distances(self, X):
        if self.metric == 'euclidean':
            return euclidean_distance(X, self.X_train)
        elif self.metric == 'manhattan':
            return manhattan_distance(X, self.X_train)
        else:
            raise ValueError(f"Invalid metric: {self.metric}")
        
    def _find_k_neighbors(self, distances):
        """
        Find k nearest neighbors. Returns indices.
        
        Parameters
        ----------
        distances : ndarray
            Shape (n_test, n_train) or (n_train,) for single sample
            
        Returns
        -------
        ndarray
            Indices of k nearest neighbors. Shape (n_test, n_neighbors) or (n_neighbors,)
        """
        # Handle single sample (1D distances)
        if distances.ndim == 1:
            return np.argsort(distances)[:self.n_neighbors]
        
        # Handle multiple samples (2D distances)
        # Sort along axis=1 (for each test sample), then take first k columns
        return np.argsort(distances, axis=1)[:, :self.n_neighbors]
    
    def predict(self, X):
        """
        Predict class labels for samples in X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples
            
        Returns
        -------
        ndarray of shape (n_samples,)
            Predicted class labels
        """
        # Check fitted
        self._check_fitted()
        
        # Convert to numpy array
        X = np.asarray(X)
        
        # Handle single sample
        if X.ndim == 1:
            X = X.reshape(1, -1)
            single_sample = True
        else:
            single_sample = False
        
        # Get distances (shape: n_test × n_train)
        distances = self._get_distances(X)
        
        # Find k nearest neighbors indices (shape: n_test × n_neighbors)
        neighbors_indices = self._find_k_neighbors(distances)
        
        # Get labels of k neighbors (shape: n_test × n_neighbors)
        k_labels = self.y_train[neighbors_indices]
        
        # Vote for most common class for each test sample
        predictions = []
        for labels in k_labels:
            # Count occurrences of each class
            unique_labels, counts = np.unique(labels, return_counts=True)
            # Get the class with highest count (if tie, first one wins)
            most_common = unique_labels[np.argmax(counts)]
            predictions.append(most_common)
        
        predictions = np.array(predictions)
        
        # Return scalar if single sample was provided
        if single_sample:
            return predictions[0]
        
        return predictions

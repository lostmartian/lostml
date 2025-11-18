import numpy as np

def euclidean_distance(x1, x2):
    """
    Compute Euclidean distance between points or arrays.
    
    Parameters
    ----------
    x1 : array-like
        First point(s). Shape: (n_features,) or (n_samples, n_features)
    x2 : array-like
        Second point(s). Shape: (n_features,) or (n_samples, n_features)
        
    Returns
    -------
    float or ndarray
        Euclidean distance(s). If both inputs are 1D, returns scalar.
        If inputs are 2D, returns array of distances.
    """
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    
    # Handle single point to single point
    if x1.ndim == 1 and x2.ndim == 1:
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
    # Handle array to array (vectorized)
    # x1: (n_test, n_features), x2: (n_train, n_features)
    # Returns: (n_test, n_train) distance matrix
    return np.sqrt(((x1[:, np.newaxis, :] - x2[np.newaxis, :, :]) ** 2).sum(axis=2))

def manhattan_distance(x1, x2):
    """
    Compute Manhattan (L1) distance between points or arrays.
    
    Parameters
    ----------
    x1 : array-like
        First point(s)
    x2 : array-like
        Second point(s)
        
    Returns
    -------
    float or ndarray
        Manhattan distance(s)
    """
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    
    # Handle single point to single point
    if x1.ndim == 1 and x2.ndim == 1:
        return np.sum(np.abs(x1 - x2))
    
    # Handle array to array (vectorized)
    return np.abs(x1[:, np.newaxis, :] - x2[np.newaxis, :, :]).sum(axis=2)
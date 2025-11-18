import numpy as np

class DecisionTree:
    """
    Decision Tree for classification and regression.
    
    Parameters
    ----------
    max_depth : int, default=None
        Maximum depth of the tree. If None, tree grows until all leaves are pure.
    min_samples_split : int, default=2
        Minimum number of samples required to split a node.
    min_samples_leaf : int, default=1
        Minimum number of samples required in a leaf node.
    criterion : str, default='gini'
        Splitting criterion. 'gini' for classification, 'mse' for regression.
    """
    
    class Node:
        """
        Internal node class to represent tree nodes.
        """
        def __init__(self):
            self.feature = None      # Index of feature to split on
            self.threshold = None    # Threshold value for splitting
            self.left = None         # Left child node
            self.right = None        # Right child node
            self.value = None        # Prediction value (for leaf nodes)
            self.is_leaf = False     # Whether this is a leaf node
    
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1, criterion='gini'):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.root = None
        self.n_classes_ = None  # For classification
        self.n_features_ = None
    
    def fit(self, X, y):
        """
        Build the decision tree from training data.
        
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
        
        if X.shape[0] == 0:
            raise ValueError("X cannot be empty")
        
        if self.criterion not in ['gini', 'mse']:
            raise ValueError(f"criterion must be 'gini' or 'mse', got '{self.criterion}'")
        
        # Store metadata
        self.n_features_ = X.shape[1]
        
        # Determine if classification or regression
        if self.criterion == 'gini':
            # Classification
            self.n_classes_ = len(np.unique(y))
        else:
            # Regression
            self.n_classes_ = None
        
        # Build the tree recursively
        self.root = self._build_tree(X, y, depth=0)
    
    def _build_tree(self, X, y, depth):
        """
        Recursively build the decision tree.
        
        Parameters
        ----------
        X : ndarray
            Training data for current node
        y : ndarray
            Target values for current node
        depth : int
            Current depth of the tree
            
        Returns
        -------
        Node
            Root node of the subtree
        """
        # Stopping condition
        if self._is_leaf(X, y, depth):
            node = self.Node()
            node.is_leaf = True
            node.value = self._make_leaf_value(y)
            return node
        
        # Find best split
        best_feature, best_threshold, best_gain = self._find_best_split(X, y)
        
        # If no valid split found (best_feature is None or gain <= 0), make leaf
        if best_feature is None or best_gain <= 0:
            node = self.Node()
            node.is_leaf = True
            node.value = self._make_leaf_value(y)
            return node
        
        # Split data into left and right based on threshold
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        
        X_left = X[left_mask]
        y_left = y[left_mask]
        X_right = X[right_mask]
        y_right = y[right_mask]

        # Check min_samples_leaf constraint
        if len(y_left) < self.min_samples_leaf or len(y_right) < self.min_samples_leaf:
            # Can't split due to min_samples_leaf constraint
            node = self.Node()
            node.is_leaf = True
            node.value = self._make_leaf_value(y)
            return node

        # Create Node
        node = self.Node()
        node.feature = best_feature
        node.threshold = best_threshold

        # Recursively build left and right subtrees
        node.left = self._build_tree(X_left, y_left, depth + 1)
        node.right = self._build_tree(X_right, y_right, depth + 1)

        return node
    
    def _is_leaf(self, X, y, depth):
        """
        Check if we should stop splitting and create a leaf node.
        
        Returns
        -------
        bool
            True if we should create a leaf node
        """
        # Stopping conditions
        # 1. Reached max_depth (if set)
        if self.max_depth is not None and depth >= self.max_depth:
            return True
        # 2. Too few samples to split
        if len(y) < self.min_samples_split:
            return True
        # 3. All samples have same label (pure node)
        if len(np.unique(y)) == 1:
            return True
        # 4. Impurity is zero (redundant with #3, but good to check)
        if len(y) == 0:
            return True
        if self._calculate_impurity(y) == 0: 
            return True
        return False
    
    def _find_best_split(self, X, y):
        """
        Find the best feature and threshold to split on.
        
        Returns
        -------
        best_feature : int or None
            Index of best feature to split on
        best_threshold : float or None
            Best threshold value
        best_gain : float
            Information gain from best split
        """

        best_feature = None
        best_threshold = None
        best_gain = -np.inf

        for features_idx in range(X.shape[1]):

            feature_values = np.unique(X[:, features_idx]) # Get unique values of the feature

            for threshold in feature_values:
                # Split data into left and right based on threshold
                left_mask = X[:, features_idx] <= threshold
                right_mask = ~left_mask
                
                # Check if split is valid (both sides have samples and meet min_samples_leaf)
                n_left = np.sum(left_mask)
                n_right = np.sum(right_mask)
                
                if n_left >= self.min_samples_leaf and n_right >= self.min_samples_leaf:
                    # Get labels for left and right splits
                    left_y = y[left_mask]
                    right_y = y[right_mask]
                    # Calculate information gain for the split
                    gain = self._calculate_information_gain(y, left_y, right_y)

                    # Update best split if current gain is better
                    if gain > best_gain:
                        best_feature = features_idx
                        best_threshold = threshold
                        best_gain = gain

        return best_feature, best_threshold, best_gain
    
    def _calculate_impurity(self, y):
        """
        Calculate impurity of labels.
        
        For classification: Gini impurity
        For regression: Variance (MSE)
        """
        if len(y) == 0:
            return 0
        
        if self.criterion == 'gini':
            # Gini impurity = 1 - sum(p_i^2) where p_i is proportion of class i
            unique, counts = np.unique(y, return_counts=True)
            proportions = counts / len(y)
            return 1 - np.sum(proportions ** 2)
        else:
            # Variance (MSE) = mean((y_i - mean(y))^2)
            return np.var(y)
    
    def _calculate_information_gain(self, y, y_left, y_right):
        """
        Calculate information gain from a split.
        
        Information Gain = Impurity(parent) - Weighted Average Impurity(children)
        """
        if len(y) == 0:
            return 0
        
        parent_impurity = self._calculate_impurity(y)
        left_impurity = self._calculate_impurity(y_left)
        right_impurity = self._calculate_impurity(y_right)
        
        # Weighted average of child impurities
        n = len(y)
        n_left = len(y_left)
        n_right = len(y_right)
        
        weighted_impurity = (n_left / n) * left_impurity + (n_right / n) * right_impurity
        
        return parent_impurity - weighted_impurity
    
    def _make_leaf_value(self, y):
        """
        Create prediction value for a leaf node.
        
        Classification: Most common class (mode)
        Regression: Mean value
        """
        if len(y) == 0:
            return 0  # Default value if empty
        
        if self.criterion == 'gini':
            # Return most common class
            unique, counts = np.unique(y, return_counts=True)
            return unique[np.argmax(counts)]
        else:
            # Return mean for regression
            return np.mean(y)
    
    def predict(self, X):
        """
        Predict target values for samples in X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples
            
        Returns
        -------
        ndarray of shape (n_samples,)
            Predicted values
        """
        X = np.asarray(X)
        
        if self.root is None:
            raise ValueError("Tree has not been fitted. Call 'fit' first.")
        
        if X.shape[0] == 0:
            return np.array([])
        
        # Validate number of features
        if X.shape[1] != self.n_features_:
            raise ValueError(
                f"X has {X.shape[1]} features, but tree expects {self.n_features_} features"
            )
        
        # For each sample, traverse tree and predict
        predictions = []
        for sample in X:
            pred = self._predict_sample(sample, self.root)
            predictions.append(pred)
        
        return np.array(predictions)
    
    def _predict_sample(self, x, node):
        """
        Predict a single sample by traversing the tree.
        
        Parameters
        ----------
        x : ndarray
            Single sample
        node : Node
            Current node in tree
            
        Returns
        -------
        Prediction value
        """
        if node.is_leaf:
            return node.value
        else:
            if x[node.feature] <= node.threshold:
                return self._predict_sample(x, node.left)
            else:
                return self._predict_sample(x, node.right)
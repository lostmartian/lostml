import numpy as np
from .decision_tree import DecisionTree


class RandomForest:
    """
    Random Forest for classification and regression.
    
    An ensemble method that combines multiple decision trees using:
    - Bootstrap aggregating (bagging): Each tree trains on random subset of data
    - Random feature selection: Each split considers random subset of features
    
    Parameters
    ----------
    n_estimators : int, default=100
        Number of trees in the forest.
    max_depth : int, default=None
        Maximum depth of each tree. If None, trees grow until all leaves are pure.
    min_samples_split : int, default=2
        Minimum number of samples required to split a node in each tree.
    min_samples_leaf : int, default=1
        Minimum number of samples required in a leaf node in each tree.
    max_features : int, float, str, or None, default='sqrt'
        Number of features to consider when looking for the best split:
        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a fraction and `int(max_features * n_features)` features are considered.
        - If 'sqrt', then `max_features = sqrt(n_features)`.
        - If 'log2', then `max_features = log2(n_features)`.
        - If None, then `max_features = n_features`.
    criterion : str, default='gini'
        Splitting criterion. 'gini' for classification, 'mse' for regression.
    bootstrap : bool, default=True
        Whether to use bootstrap sampling when building trees.
    random_state : int or None, default=None
        Random seed for reproducibility.
    """
    
    def __init__(
        self,
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        criterion='gini',
        bootstrap=True,
        random_state=None
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.criterion = criterion
        self.bootstrap = bootstrap
        self.random_state = random_state
        
        # Will be set during fit
        self.trees = []
        self.n_features_ = None
        self.n_classes_ = None
        self.is_classification = None
        
        # Set random seed if provided
        if self.random_state is not None:
            np.random.seed(self.random_state)
    
    def fit(self, X, y):
        """
        Build a forest of trees from training data.
        
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
        self.is_classification = (self.criterion == 'gini')
        
        if self.is_classification:
            self.n_classes_ = len(np.unique(y))
        else:
            self.n_classes_ = None
        
        # Calculate max_features value
        max_features = self._get_max_features(self.n_features_)
        
        # Initialize list to store trees
        self.trees = []
        
        # Build n_estimators trees
        for i in range(self.n_estimators):
            # Create bootstrap sample (with replacement)
            if self.bootstrap:
                n_samples = X.shape[0]
                # Random indices with replacement
                indices = np.random.choice(n_samples, size=n_samples, replace=True)
                X_bootstrap = X[indices]
                y_bootstrap = y[indices]
            else:
                # Use all data
                X_bootstrap = X
                y_bootstrap = y
            
            # Create and train a decision tree
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                criterion=self.criterion
            )
            
            # Store original method and create wrapper for random feature selection
            original_find_best_split = tree._find_best_split
            
            def create_random_feature_splitter(tree_instance, n_features_to_use):
                """Create a function that restricts feature search to random subset."""
                def random_feature_find_best_split(X_subset, y_subset):
                    """Wrapper that restricts feature search to random subset."""
                    # Select random subset of features for this split
                    n_available_features = X_subset.shape[1]
                    n_to_select = min(n_features_to_use, n_available_features)
                    feature_indices = np.random.choice(
                        n_available_features,
                        size=n_to_select,
                        replace=False
                    )
                    
                    best_feature = None
                    best_threshold = None
                    best_gain = -np.inf
                    
                    # Only search through selected features
                    for feature_idx in feature_indices:
                        feature_values = np.unique(X_subset[:, feature_idx])
                        
                        for threshold in feature_values:
                            left_mask = X_subset[:, feature_idx] <= threshold
                            right_mask = ~left_mask
                            
                            n_left = np.sum(left_mask)
                            n_right = np.sum(right_mask)
                            
                            if n_left >= tree_instance.min_samples_leaf and n_right >= tree_instance.min_samples_leaf:
                                left_y = y_subset[left_mask]
                                right_y = y_subset[right_mask]
                                gain = tree_instance._calculate_information_gain(y_subset, left_y, right_y)
                                
                                if gain > best_gain:
                                    best_feature = feature_idx
                                    best_threshold = threshold
                                    best_gain = gain
                    
                    return best_feature, best_threshold, best_gain
                
                return random_feature_find_best_split
            
            # Replace the method with random feature version
            tree._find_best_split = create_random_feature_splitter(tree, max_features)
            
            # Train the tree
            tree.fit(X_bootstrap, y_bootstrap)
            
            # Store the tree
            self.trees.append(tree)
    
    def _get_max_features(self, n_features):
        """
        Calculate the number of features to consider at each split.
        
        Parameters
        ----------
        n_features : int
            Total number of features
            
        Returns
        -------
        int
            Number of features to use
        """
        if self.max_features is None:
            return n_features
        elif isinstance(self.max_features, int):
            return min(self.max_features, n_features)
        elif isinstance(self.max_features, float):
            return max(1, int(self.max_features * n_features))
        elif self.max_features == 'sqrt':
            return int(np.sqrt(n_features))
        elif self.max_features == 'log2':
            return int(np.log2(n_features))
        else:
            raise ValueError(
                f"max_features must be int, float, 'sqrt', 'log2', or None, "
                f"got '{self.max_features}'"
            )
    
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
        
        if len(self.trees) == 0:
            raise ValueError("Forest has not been fitted. Call 'fit' first.")
        
        if X.shape[0] == 0:
            return np.array([])
        
        # Validate number of features
        if X.shape[1] != self.n_features_:
            raise ValueError(
                f"X has {X.shape[1]} features, but forest expects {self.n_features_} features"
            )
        
        # Get predictions from all trees
        all_predictions = []
        for tree in self.trees:
            predictions = tree.predict(X)
            all_predictions.append(predictions)
        
        # Stack predictions: shape (n_estimators, n_samples)
        all_predictions = np.array(all_predictions)
        
        # Aggregate predictions
        if self.is_classification:
            # Classification: Majority voting
            # For each sample, find the most common prediction across all trees
            final_predictions = []
            for sample_idx in range(X.shape[0]):
                # Get predictions from all trees for this sample
                sample_predictions = all_predictions[:, sample_idx]
                # Find most common class (mode)
                unique, counts = np.unique(sample_predictions, return_counts=True)
                most_common = unique[np.argmax(counts)]
                final_predictions.append(most_common)
            return np.array(final_predictions)
        else:
            # Regression: Average predictions
            return np.mean(all_predictions, axis=0)
    
    def predict_proba(self, X):
        """
        Predict class probabilities for classification.
        
        Only available for classification (criterion='gini').
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples
            
        Returns
        -------
        ndarray of shape (n_samples, n_classes)
            Class probabilities
        """
        if not self.is_classification:
            raise ValueError("predict_proba is only available for classification")
        
        X = np.asarray(X)
        
        if len(self.trees) == 0:
            raise ValueError("Forest has not been fitted. Call 'fit' first.")
        
        if X.shape[1] != self.n_features_:
            raise ValueError(
                f"X has {X.shape[1]} features, but forest expects {self.n_features_} features"
            )
        
        # Get predictions from all trees
        all_predictions = []
        for tree in self.trees:
            predictions = tree.predict(X)
            all_predictions.append(predictions)
        
        # Stack predictions: shape (n_estimators, n_samples)
        all_predictions = np.array(all_predictions)
        
        # Calculate probabilities for each sample
        n_samples = X.shape[0]
        probabilities = np.zeros((n_samples, self.n_classes_))
        
        for sample_idx in range(n_samples):
            # Get predictions from all trees for this sample
            sample_predictions = all_predictions[:, sample_idx]
            # Count votes for each class
            for class_idx in range(self.n_classes_):
                # Count how many trees predicted this class
                votes = np.sum(sample_predictions == class_idx)
                probabilities[sample_idx, class_idx] = votes / self.n_estimators
        
        return probabilities


Random Forest
=============

Random Forest is an ensemble method that combines multiple decision trees using bootstrap aggregating (bagging) and random feature selection. It reduces overfitting and improves generalization compared to a single decision tree.

RandomForest Class
------------------

.. autoclass:: lostml.tree.random_forest.RandomForest
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Parameters
----------

- ``n_estimators``: Number of trees in the forest (default: 100)
- ``max_depth``: Maximum depth of each tree. If None, trees grow until all leaves are pure (default: None)
- ``min_samples_split``: Minimum number of samples required to split a node in each tree (default: 2)
- ``min_samples_leaf``: Minimum number of samples required in a leaf node in each tree (default: 1)
- ``max_features``: Number of features to consider when looking for the best split (default: 'sqrt')
  - ``'sqrt'``: sqrt(n_features) features
  - ``'log2'``: log2(n_features) features
  - ``None``: All features
  - ``int``: Exact number of features
  - ``float``: Fraction of features (e.g., 0.5 = 50% of features)
- ``criterion``: Splitting criterion
  - ``'gini'``: Gini impurity for classification
  - ``'mse'``: Mean squared error (variance) for regression
- ``bootstrap``: Whether to use bootstrap sampling when building trees (default: True)
- ``random_state``: Random seed for reproducibility (default: None)

Examples
--------

Classification
~~~~~~~~~~~~~~

.. code-block:: python

   from lostml.tree import RandomForest
   import numpy as np

   X = np.array([[1, 2], [2, 3], [3, 4], [5, 6], [6, 7], [7, 8]])
   y = np.array([0, 0, 0, 1, 1, 1])
   
   rf = RandomForest(n_estimators=100, criterion='gini', random_state=42)
   rf.fit(X, y)
   predictions = rf.predict(X)
   
   # Get class probabilities
   probabilities = rf.predict_proba(X)

Regression
~~~~~~~~~

.. code-block:: python

   from lostml.tree import RandomForest
   import numpy as np

   X = np.array([[1], [2], [3], [5], [6], [7]])
   y = np.array([2, 4, 6, 10, 12, 14])
   
   rf = RandomForest(n_estimators=100, criterion='mse', random_state=42)
   rf.fit(X, y)
   predictions = rf.predict(X)

Customizing the Forest
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Control tree complexity and feature selection
   rf = RandomForest(
       n_estimators=200,        # More trees = better but slower
       max_depth=10,            # Limit depth of each tree
       max_features='sqrt',     # Use sqrt(n_features) at each split
       min_samples_split=5,    # Require more samples to split
       min_samples_leaf=2,     # Ensure leaves have enough samples
       bootstrap=True,          # Use bootstrap sampling
       random_state=42          # For reproducibility
   )

How It Works
------------

1. **Bootstrap Sampling**: Each tree is trained on a random subset of data (with replacement)
2. **Random Feature Selection**: At each split, only a random subset of features is considered
3. **Tree Building**: Each tree is built independently using the DecisionTree algorithm
4. **Ensemble Prediction**: 
   - **Classification**: Majority voting across all trees
   - **Regression**: Average of predictions from all trees

Key Concepts
------------

**Bootstrap Aggregating (Bagging)**
   - Each tree sees a different random sample of the training data
   - About 63% of samples appear in each bootstrap sample
   - Reduces variance and overfitting

**Random Feature Selection**
   - At each split, only a subset of features is considered
   - Increases diversity among trees
   - Default 'sqrt' is a good balance between diversity and performance

**Ensemble Voting**
   - Classification: Most common prediction wins (majority vote)
   - Regression: Average of all tree predictions
   - More trees = more stable predictions

Advantages
----------

- **Reduces Overfitting**: Ensemble of trees is less prone to overfitting than a single tree
- **Handles Non-linearity**: Can capture complex non-linear relationships
- **Feature Importance**: Can identify important features (via tree splits)
- **Robust**: Less sensitive to outliers and noise
- **No Feature Scaling**: Works well without feature normalization

Tips
----

- **n_estimators**: Start with 100-200. More trees = better but slower
- **max_features**: 'sqrt' is a good default. Use 'log2' for high-dimensional data
- **max_depth**: Limit depth to prevent overfitting (default: None allows full growth)
- **bootstrap**: Keep True for better generalization (default: True)
- **random_state**: Set for reproducible results
- **Classification**: Use ``predict_proba()`` to get class probabilities
- **Regression**: Predictions are averaged across all trees

Comparison with Decision Tree
------------------------------

- **Decision Tree**: Single tree, can overfit easily, fast training
- **Random Forest**: Ensemble of trees, reduces overfitting, slower training but better generalization

Use Random Forest when:
- You want better accuracy than a single decision tree
- You have enough data and computational resources
- You need robust predictions that handle noise well


Decision Tree
=============

Decision Tree is a tree-based algorithm that recursively splits data to make predictions. It supports both classification and regression tasks.

DecisionTree Class
------------------

.. autoclass:: lostml.tree.decision_tree.DecisionTree
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Parameters
----------

- ``max_depth``: Maximum depth of the tree. If None, tree grows until all leaves are pure (default: None)
- ``min_samples_split``: Minimum number of samples required to split a node (default: 2)
- ``min_samples_leaf``: Minimum number of samples required in a leaf node (default: 1)
- ``criterion``: Splitting criterion
  - ``'gini'``: Gini impurity for classification
  - ``'mse'``: Mean squared error (variance) for regression

Examples
--------

Classification
~~~~~~~~~~~~~~

.. code-block:: python

   from lostml.tree import DecisionTree
   import numpy as np

   X = np.array([[1, 2], [2, 3], [3, 4], [5, 6], [6, 7]])
   y = np.array([0, 0, 1, 1, 1])
   
   tree = DecisionTree(max_depth=5, criterion='gini')
   tree.fit(X, y)
   predictions = tree.predict(X)

Regression
~~~~~~~~~~

.. code-block:: python

   from lostml.tree import DecisionTree
   import numpy as np

   X = np.array([[1], [2], [3], [5], [6]])
   y = np.array([2, 4, 6, 10, 12])
   
   tree = DecisionTree(max_depth=3, criterion='mse')
   tree.fit(X, y)
   predictions = tree.predict(X)

Controlling Overfitting
~~~~~~~~~~~~~~~~~~~~~~~

Decision trees can easily overfit. Use these parameters to control tree growth:

.. code-block:: python

   # Prevent overfitting with constraints
   tree = DecisionTree(
       max_depth=5,              # Limit tree depth
       min_samples_split=10,     # Require more samples to split
       min_samples_leaf=5        # Ensure leaves have enough samples
   )

How It Works
------------

1. **Splitting**: At each node, the algorithm finds the best feature and threshold to split on
2. **Impurity**: Uses Gini impurity (classification) or variance (regression) to measure node purity
3. **Information Gain**: Chooses splits that maximize information gain
4. **Recursion**: Recursively builds left and right subtrees
5. **Stopping**: Stops when stopping conditions are met (max_depth, min_samples, pure nodes)
6. **Prediction**: Traverses tree from root to leaf, returns leaf value

Tips
----

- **max_depth**: Start with 5-10. Too deep = overfitting, too shallow = underfitting
- **min_samples_split**: Higher values = simpler trees (default: 2)
- **min_samples_leaf**: Higher values = more conservative splits (default: 1)
- **criterion**: Use 'gini' for classification, 'mse' for regression
- **Interpretability**: Decision trees are highly interpretable - you can visualize the decision path


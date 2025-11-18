Quick Start Guide
=================

This guide will get you up and running with lostml in minutes. We'll cover the main algorithms with practical examples.

Linear Regression
-----------------

Linear regression predicts continuous values using a linear relationship between features and target.

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   from lostml import LinearRegression
   import numpy as np

   # Create sample data
   X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
   y = np.array([2, 3, 4, 5])

   # Initialize model
   model = LinearRegression(learning_rate=0.01, n_iterations=1000)
   
   # Train the model
   model.fit(X, y)
   
   # Make predictions
   predictions = model.predict(X)
   print(predictions)

Parameters:
   - ``learning_rate``: Step size for gradient descent (default: 0.01)
   - ``n_iterations``: Number of training iterations (default: 1000)

Ridge Regression
----------------

Ridge regression adds L2 regularization to prevent overfitting by penalizing large weights.

.. code-block:: python

   from lostml import RigdeRegression
   import numpy as np

   X = np.array([[1, 2], [2, 3], [3, 4], [5, 6]])
   y = np.array([2, 3, 4, 6])

   # Initialize with regularization parameter
   model = RigdeRegression(
       lambda_param=0.1,      # Regularization strength
       learning_rate=0.01,
       n_iterations=1000
   )
   model.fit(X, y)
   predictions = model.predict(X)

The ``lambda_param`` controls regularization strength—higher values mean more regularization.

Lasso Regression
----------------

Lasso regression uses L1 regularization, which can set some coefficients to exactly zero (feature selection).

.. code-block:: python

   from lostml import LassoRegression
   import numpy as np

   X = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
   y = np.array([2, 3, 4])

   model = LassoRegression(
       lambda_param=0.1,
       learning_rate=0.01,
       n_iterations=1000
   )
   model.fit(X, y)
   predictions = model.predict(X)

Lasso is useful when you have many features and want automatic feature selection.

Elastic Net
-----------

Elastic Net combines both L1 and L2 regularization, getting benefits from both.

.. code-block:: python

   from lostml import ElasticNet
   import numpy as np

   X = np.array([[1, 2], [2, 3], [3, 4]])
   y = np.array([2, 3, 4])

   model = ElasticNet(
       alpha=0.1,           # Overall regularization strength
       l1_ratio=0.5,        # Mix of L1 vs L2 (0.5 = equal mix)
       learning_rate=0.01,
       n_iterations=1000
   )
   model.fit(X, y)
   predictions = model.predict(X)

- ``alpha``: Total regularization strength
- ``l1_ratio``: Proportion of L1 (0 = pure L2, 1 = pure L1)

Logistic Regression
-------------------

Logistic regression is used for binary classification, outputting probabilities.

Basic Classification
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from lostml import LogisticRegression
   import numpy as np

   # Binary classification data
   X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
   y = np.array([0, 0, 1, 1])  # Binary labels

   # Initialize and train
   model = LogisticRegression(learning_rate=0.01, n_iterations=1000)
   model.fit(X, y)

   # Get probabilities (0 to 1)
   probabilities = model.predict(X)
   print(f"Probabilities: {probabilities}")
   
   # Get class probabilities for both classes
   class_probs = model.predict_proba(X)
   print(f"Class probabilities shape: {class_probs.shape}")
   # Returns: [[P(class=0), P(class=1)], ...]

Getting Class Predictions
~~~~~~~~~~~~~~~~~~~~~~~~~

To convert probabilities to class predictions:

.. code-block:: python

   # Threshold at 0.5
   predictions = (probabilities >= 0.5).astype(int)
   print(predictions)

K-Nearest Neighbors (KNN)
-------------------------

KNN is a simple, instance-based learning algorithm that makes predictions based on the k nearest training examples.

Basic Classification
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from lostml.neighbors import KNN
   import numpy as np

   # Training data
   X_train = np.array([[1, 2], [2, 3], [3, 4], [5, 6], [6, 7]])
   y_train = np.array([0, 0, 1, 1, 1])

   # Test data
   X_test = np.array([[2.5, 3.5], [5.5, 6.5]])

   # Initialize KNN
   knn = KNN(n_neighbors=3, metric='euclidean')
   knn.fit(X_train, y_train)

   # Predict
   predictions = knn.predict(X_test)
   print(predictions)  # [0, 1]

Parameters:
   - ``n_neighbors``: Number of neighbors to consider (default: 5)
   - ``metric``: Distance metric - 'euclidean' or 'manhattan' (default: 'euclidean')

Distance Metrics
~~~~~~~~~~~~~~~~

KNN supports different distance metrics:

.. code-block:: python

   # Euclidean distance (default)
   knn_euclidean = KNN(n_neighbors=5, metric='euclidean')
   
   # Manhattan distance
   knn_manhattan = KNN(n_neighbors=5, metric='manhattan')

Real-World Example
------------------

Here's a complete example using KNN on a simple 2D classification problem:

.. code-block:: python

   from lostml.neighbors import KNN
   import numpy as np

   # Create two clusters
   np.random.seed(42)
   cluster1 = np.random.randn(20, 2) + [0, 0]
   cluster2 = np.random.randn(20, 2) + [5, 5]

   X_train = np.vstack([cluster1, cluster2])
   y_train = np.array([0] * 20 + [1] * 20)

   # Test point closer to cluster 1
   X_test = np.array([[0.5, 0.5]])

   knn = KNN(n_neighbors=5)
   knn.fit(X_train, y_train)
   prediction = knn.predict(X_test)
   
   print(f"Predicted class: {prediction}")  # Should be 0

Tips and Best Practices
-----------------------

1. **Learning Rate**: Start with 0.01 and adjust. Too high = unstable, too low = slow convergence.

2. **Iterations**: 1000 is usually sufficient, but increase for complex problems.

3. **K in KNN**: 
   - Odd numbers avoid ties (3, 5, 7)
   - Too small = overfitting, too large = underfitting
   - Start with k=5

4. **Regularization**: 
   - Ridge: Use when you have many correlated features
   - Lasso: Use when you want feature selection
   - Elastic Net: Best of both worlds

5. **Data Preprocessing**: Always normalize/standardize features for best results.

Next Steps
----------

- Check out the :doc:`api/index` for detailed API documentation
- Explore the source code to understand implementations
- Try the examples with your own data

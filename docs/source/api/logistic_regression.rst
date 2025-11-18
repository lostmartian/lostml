Logistic Regression
===================

Logistic regression is a classification algorithm that uses the sigmoid function to output probabilities for binary classification.

LogisticRegression Classifier
------------------------------

.. autoclass:: lostml.linear_models.logistic_regression.LogisticRegression
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Parameters
----------

- ``learning_rate``: Step size for gradient descent (default: 0.01)
- ``n_iterations``: Number of training iterations (default: 1000)

Methods
-------

``fit(X, y)``
   Train the model on data X with binary labels y (0 or 1).

``predict(X)``
   Return probability estimates for each sample (values between 0 and 1).

``predict_proba(X)``
   Return probability estimates for both classes. Returns array of shape (n_samples, 2) where:
   - Column 0: P(class=0)
   - Column 1: P(class=1)

Examples
--------

Basic Classification
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from lostml import LogisticRegression
   import numpy as np

   # Binary classification data
   X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
   y = np.array([0, 0, 1, 1])

   model = LogisticRegression(learning_rate=0.01, n_iterations=1000)
   model.fit(X, y)

   # Get probabilities
   probabilities = model.predict(X)
   print(probabilities)  # [0.1, 0.3, 0.7, 0.9] (example)

Getting Class Predictions
~~~~~~~~~~~~~~~~~~~~~~~~~

To convert probabilities to binary predictions:

.. code-block:: python

   # Threshold at 0.5
   predictions = (probabilities >= 0.5).astype(int)
   print(predictions)  # [0, 0, 1, 1]

Getting Class Probabilities
~~~~~~~~~~~~~~~~~~~~~~~~~~~

To get probabilities for both classes:

.. code-block:: python

   class_probs = model.predict_proba(X)
   # Returns: [[0.9, 0.1], [0.7, 0.3], [0.3, 0.7], [0.1, 0.9]]
   # Format: [P(class=0), P(class=1)] for each sample

How It Works
------------

Logistic regression:

1. Computes a linear combination: ``z = X * weights + bias``
2. Applies sigmoid function: ``σ(z) = 1 / (1 + e^(-z))``
3. Outputs probabilities between 0 and 1
4. Uses cross-entropy loss for training
5. Updates weights using gradient descent

The sigmoid function ensures outputs are valid probabilities and provides a smooth, differentiable function for optimization.

Tips
----

- **Learning rate**: Start with 0.01. If loss doesn't decrease, try smaller values.
- **Iterations**: 1000 is usually sufficient. Monitor loss to see if more are needed.
- **Data**: Ensure labels are binary (0 and 1). Normalize features for best results.
- **Threshold**: 0.5 is standard, but adjust based on your problem (precision vs recall trade-off).

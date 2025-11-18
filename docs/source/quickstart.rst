Quick Start
===========

This guide will help you get started with lostml. We'll cover basic usage of both linear and logistic regression models.

Linear Regression
-----------------

Linear regression is used for predicting continuous values. Here's a simple example:

.. code-block:: python

   from lostml import LinearRegression
   import numpy as np

   # Create sample data
   X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
   y = np.array([2, 3, 4, 5])

   # Initialize and fit
   model = LinearRegression(learning_rate=0.01, n_iterations=1000)
   model.fit(X, y)

   # Predict
   predictions = model.predict(X)
   print(predictions)

Ridge Regression
----------------

Ridge regression adds L2 regularization to prevent overfitting:

.. code-block:: python

   from lostml import RigdeRegression
   import numpy as np

   X = np.array([[1, 2], [2, 3], [3, 4]])
   y = np.array([2, 3, 4])

   # Initialize with regularization parameter
   model = RigdeRegression(lambda_param=0.1, learning_rate=0.01, n_iterations=1000)
   model.fit(X, y)
   predictions = model.predict(X)

Lasso Regression
----------------

Lasso regression adds L1 regularization, which can set some coefficients to zero:

.. code-block:: python

   from lostml import LassoRegression
   import numpy as np

   X = np.array([[1, 2], [2, 3], [3, 4]])
   y = np.array([2, 3, 4])

   model = LassoRegression(lambda_param=0.1, learning_rate=0.01, n_iterations=1000)
   model.fit(X, y)
   predictions = model.predict(X)

Elastic Net
-----------

Elastic Net combines both L1 and L2 regularization:

.. code-block:: python

   from lostml import ElasticNet
   import numpy as np

   X = np.array([[1, 2], [2, 3], [3, 4]])
   y = np.array([2, 3, 4])

   model = ElasticNet(alpha=0.1, l1_ratio=0.5, learning_rate=0.01, n_iterations=1000)
   model.fit(X, y)
   predictions = model.predict(X)

Logistic Regression
-------------------

Logistic regression is used for binary classification:

.. code-block:: python

   from lostml import LogisticRegression
   import numpy as np

   # Create sample data (binary classification)
   X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
   y = np.array([0, 0, 1, 1])  # Binary labels

   # Initialize and fit
   model = LogisticRegression(learning_rate=0.01, n_iterations=1000)
   model.fit(X, y)

   # Get probabilities
   probabilities = model.predict(X)
   print(f"Probabilities: {probabilities}")
   
   # Get class probabilities for both classes
   class_probs = model.predict_proba(X)
   print(f"Class probabilities: {class_probs}")

Next Steps
----------

For more detailed information, see the :doc:`api/index` section.


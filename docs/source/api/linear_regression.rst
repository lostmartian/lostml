Linear Regression Models
========================

This module contains implementations of various linear regression models, all using gradient descent for optimization.

LinearRegression
----------------

Standard linear regression using gradient descent. Fits a linear model to minimize mean squared error.

.. autoclass:: lostml.linear_models.linear_regression.LinearRegression
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Example
~~~~~~~

.. code-block:: python

   from lostml import LinearRegression
   import numpy as np

   X = np.array([[1, 2], [2, 3], [3, 4]])
   y = np.array([2, 3, 4])
   
   model = LinearRegression(learning_rate=0.01, n_iterations=1000)
   model.fit(X, y)
   predictions = model.predict(X)

Ridge Regression
----------------

Ridge regression adds L2 regularization (penalty on squared weights) to prevent overfitting.

.. autoclass:: lostml.linear_models.linear_regression.RigdeRegression
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Parameters
~~~~~~~~~~

- ``lambda_param``: Regularization strength. Higher values = more regularization (default: 1.0)

Example
~~~~~~~

.. code-block:: python

   from lostml import RigdeRegression
   import numpy as np

   X = np.array([[1, 2], [2, 3], [3, 4]])
   y = np.array([2, 3, 4])
   
   model = RigdeRegression(lambda_param=0.1, learning_rate=0.01, n_iterations=1000)
   model.fit(X, y)
   predictions = model.predict(X)

When to Use
~~~~~~~~~~~

- When you have many features
- When features are correlated
- To prevent overfitting

Lasso Regression
----------------

Lasso regression uses L1 regularization (penalty on absolute weights), which can set coefficients to exactly zero (feature selection).

.. autoclass:: lostml.linear_models.linear_regression.LassoRegression
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Parameters
~~~~~~~~~~

- ``lambda_param``: Regularization strength. Higher values = more features set to zero (default: 1.0)

Example
~~~~~~~

.. code-block:: python

   from lostml import LassoRegression
   import numpy as np

   X = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
   y = np.array([2, 3, 4])
   
   model = LassoRegression(lambda_param=0.1, learning_rate=0.01, n_iterations=1000)
   model.fit(X, y)
   # Some weights may be exactly zero
   print(model.weights)

When to Use
~~~~~~~~~~~

- When you want automatic feature selection
- When you have many irrelevant features
- To create sparse models

Elastic Net
-----------

Elastic Net combines both L1 and L2 regularization, getting benefits from both approaches.

.. autoclass:: lostml.linear_models.linear_regression.ElasticNet
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Parameters
~~~~~~~~~~

- ``alpha``: Overall regularization strength (default: 1.0)
- ``l1_ratio``: Proportion of L1 vs L2 regularization (default: 0.5)
  - 0.0 = pure Ridge (L2 only)
  - 1.0 = pure Lasso (L1 only)
  - 0.5 = equal mix

Example
~~~~~~~

.. code-block:: python

   from lostml import ElasticNet
   import numpy as np

   X = np.array([[1, 2], [2, 3], [3, 4]])
   y = np.array([2, 3, 4])
   
   model = ElasticNet(alpha=0.1, l1_ratio=0.5, learning_rate=0.01, n_iterations=1000)
   model.fit(X, y)
   predictions = model.predict(X)

When to Use
~~~~~~~~~~~

- When you want benefits of both Ridge and Lasso
- When you have groups of correlated features
- As a general-purpose regularized regression

Common Parameters
-----------------

All regression models share these parameters:

- ``learning_rate``: Step size for gradient descent (default: 0.01)
  - Too high: may overshoot and diverge
  - Too low: slow convergence
  
- ``n_iterations``: Number of gradient descent iterations (default: 1000)
  - More iterations = better convergence (up to a point)
  - Monitor convergence to find optimal value

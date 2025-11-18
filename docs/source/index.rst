lostml Documentation
====================

Welcome to **lostml** - a from-scratch machine learning library built for learning and understanding ML algorithms from the ground up.

.. note::

   lostml is designed to be educational and transparent. Every algorithm is implemented from scratch with clear, readable code—no black boxes.

What is lostml?
---------------

lostml is a Python machine learning library that implements core ML algorithms from scratch. Unlike libraries that hide implementation details, lostml provides:

- **Transparent implementations** - See exactly how each algorithm works
- **Educational focus** - Learn ML by understanding the code
- **Clean, readable code** - Well-documented and easy to follow
- **From scratch** - No reliance on high-level ML frameworks for core algorithms

Features
--------

**Linear Models**
   - Linear Regression
   - Ridge Regression (L2 regularization)
   - Lasso Regression (L1 regularization)
   - Elastic Net (L1 + L2 regularization)

**Classification**
   - Logistic Regression
   - K-Nearest Neighbors (KNN)

**Utilities**
   - Distance metrics (Euclidean, Manhattan)
   - Evaluation metrics (coming soon)

Quick Example
-------------

Here's a quick taste of what lostml can do:

.. code-block:: python

   from lostml import LinearRegression
   from lostml.neighbors import KNN
   import numpy as np

   # Linear Regression
   X = np.array([[1, 2], [2, 3], [3, 4]])
   y = np.array([2, 3, 4])
   model = LinearRegression()
   model.fit(X, y)
   predictions = model.predict(X)

   # K-Nearest Neighbors
   knn = KNN(n_neighbors=3)
   knn.fit(X_train, y_train)
   predictions = knn.predict(X_test)

Documentation Contents
----------------------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started:

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: API Reference:

   api/index

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

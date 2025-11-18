K-Nearest Neighbors
===================

K-Nearest Neighbors (KNN) is an instance-based learning algorithm that makes predictions based on the k nearest training examples.

KNN Classifier
--------------

.. autoclass:: lostml.neighbors.knn.KNN
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Examples
--------

Basic Classification
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from lostml.neighbors import KNN
   import numpy as np

   X_train = np.array([[1, 2], [2, 3], [3, 4], [5, 6]])
   y_train = np.array([0, 0, 1, 1])
   
   knn = KNN(n_neighbors=3)
   knn.fit(X_train, y_train)
   predictions = knn.predict(np.array([[2.5, 3.5]]))

Using Different Metrics
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Euclidean distance (default)
   knn = KNN(n_neighbors=5, metric='euclidean')
   
   # Manhattan distance
   knn = KNN(n_neighbors=5, metric='manhattan')


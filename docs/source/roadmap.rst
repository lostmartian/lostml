Algorithm Roadmap
=================

This page tracks the implementation status of algorithms in lostml.

Implemented Algorithms ✅
--------------------------

Linear Models
~~~~~~~~~~~~~

- ✅ **Linear Regression** - Basic linear regression using gradient descent
- ✅ **Ridge Regression** - Linear regression with L2 regularization
- ✅ **Lasso Regression** - Linear regression with L1 regularization (feature selection)
- ✅ **Elastic Net** - Combines L1 and L2 regularization

Classification
~~~~~~~~~~~~~

- ✅ **Logistic Regression** - Binary classification using sigmoid function
- ✅ **K-Nearest Neighbors (KNN)** - Instance-based classification with distance metrics

Tree-Based Models
~~~~~~~~~~~~~~~~~

- ✅ **Decision Tree** - Classification and regression using recursive splitting
  - *Status*: Implemented
  - *Use Cases*: Interpretable models, feature importance, non-linear relationships
  - *Features*: Supports both Gini (classification) and MSE (regression) criteria

- ✅ **Random Forest** - Ensemble of decision trees with bootstrap aggregating
  - *Status*: Implemented
  - *Use Cases*: General-purpose classification and regression, handles overfitting well
  - *Features*: Bootstrap sampling, random feature selection, majority voting (classification), averaging (regression)

Utilities
~~~~~~~~~

- ✅ **Distance Metrics** - Euclidean and Manhattan distance functions

Planned Algorithms 🚧
---------------------

Unsupervised Learning
~~~~~~~~~~~~~~~~~~~~~

- ⏳ **K-Means Clustering** - Partition data into k clusters
  - *Status*: Planned
  - *Priority*: High
  - *Use Cases*: Customer segmentation, data exploration, pattern discovery

- ⏳ **PCA (Principal Component Analysis)** - Dimensionality reduction
  - *Status*: Planned
  - *Priority*: High
  - *Use Cases*: Feature reduction, visualization, noise reduction

Additional Algorithms
~~~~~~~~~~~~~~~~~~~~~

- ⏳ **Naive Bayes** - Probabilistic classifier
  - *Status*: Planned
  - *Priority*: Medium
  - *Use Cases*: Text classification, spam detection, fast classification

- ⏳ **Support Vector Machine (SVM)** - Maximum margin classifier
  - *Status*: Planned
  - *Priority*: Medium
  - *Use Cases*: Classification with clear margins, non-linear data (with kernels)

Implementation Status
----------------------

**Current Progress**: 8/12 algorithms implemented (67%)

**Next Up**: K-Means → PCA → Naive Bayes

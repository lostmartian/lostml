# lostml

> A from-scratch machine learning library built for learning and understanding ML algorithms from the ground up.

## 🚀 Quick Start

```python
from lostml import LinearRegression, LogisticRegression
from lostml.neighbors import KNN
from lostml.tree import DecisionTree, RandomForest
import numpy as np

# Linear Regression
X = np.array([[1, 2], [2, 3], [3, 4]])
y = np.array([2, 3, 4])
model = LinearRegression(learning_rate=0.01, n_iterations=1000)
model.fit(X, y)
predictions = model.predict(X)

# Decision Tree
tree = DecisionTree(max_depth=5, criterion='gini')
tree.fit(X_train, y_train)
predictions = tree.predict(X_test)

# Random Forest
rf = RandomForest(n_estimators=100, criterion='gini', random_state=42)
rf.fit(X_train, y_train)
predictions = rf.predict(X_test)

# K-Nearest Neighbors
knn = KNN(n_neighbors=5, metric='euclidean')
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)
```

## 📦 Installation

```bash
git clone https://github.com/yourusername/lostml.git
cd lostml
pip install -e .
```

## ✨ What's Inside

### Implemented ✅

**Linear Models**
- ✅ Linear Regression
- ✅ Ridge Regression (L2 regularization)
- ✅ Lasso Regression (L1 regularization)
- ✅ Elastic Net (L1 + L2 regularization)

**Classification**
- ✅ Logistic Regression
- ✅ K-Nearest Neighbors (KNN)

**Tree-Based Models**
- ✅ Decision Tree (Classification & Regression)
- ✅ Random Forest (Classification & Regression)

**Utilities**
- ✅ Distance metrics (Euclidean, Manhattan)

### Coming Soon 🚧

**Unsupervised Learning**
- ⏳ K-Means Clustering
- ⏳ PCA (Principal Component Analysis)

**Additional Algorithms**
- ⏳ Naive Bayes
- ⏳ Support Vector Machine (SVM)

## 📚 Documentation

Full documentation with examples and API reference: **[View Docs](https://lostml.sahilgangurde.me)**

## 🧪 Testing

```bash
pip install pytest
pytest
```

## 🛠️ Requirements

- Python 3.7+
- NumPy

## 🎯 Why lostml?

Built from scratch to understand the inner workings of machine learning algorithms. No black boxes—just clean, readable implementations.

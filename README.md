# lostml

> A from-scratch machine learning library built for learning and understanding ML algorithms from the ground up.

## 🚀 Quick Start

```python
from lostml import LinearRegression, LogisticRegression
from lostml.neighbors import KNN
import numpy as np

# Linear Regression
X = np.array([[1, 2], [2, 3], [3, 4]])
y = np.array([2, 3, 4])
model = LinearRegression(learning_rate=0.01, n_iterations=1000)
model.fit(X, y)
predictions = model.predict(X)

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

- **Linear Models**: Linear Regression, Ridge, Lasso, Elastic Net
- **Classification**: Logistic Regression, K-Nearest Neighbors  
- **Utilities**: Distance metrics, evaluation metrics

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

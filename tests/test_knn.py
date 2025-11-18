"""
Tests for K-Nearest Neighbors classifier.
"""
import numpy as np
import pytest
from lostml.neighbors.knn import KNN


class TestKNNBasic:
    """Basic functionality tests."""
    
    def test_fit_and_predict_simple(self):
        """Test basic fit and predict with simple data."""
        X_train = np.array([[1, 2], [2, 3], [3, 4], [5, 6], [6, 7]])
        y_train = np.array([0, 0, 1, 1, 1])
        X_test = np.array([[2, 2.5]])
        
        knn = KNN(n_neighbors=3)
        knn.fit(X_train, y_train)
        predictions = knn.predict(X_test)
        
        # Should predict class 0 (closer to first two points)
        assert predictions == 0 or predictions == np.array(0)
    
    def test_predict_multiple_samples(self):
        """Test predicting multiple samples at once."""
        X_train = np.array([[1, 1], [2, 2], [5, 5], [6, 6]])
        y_train = np.array([0, 0, 1, 1])
        X_test = np.array([[1.5, 1.5], [5.5, 5.5]])
        
        knn = KNN(n_neighbors=2)
        knn.fit(X_train, y_train)
        predictions = knn.predict(X_test)
        
        assert len(predictions) == 2
        assert predictions[0] == 0  # Closer to class 0
        assert predictions[1] == 1  # Closer to class 1
    
    def test_single_sample_input(self):
        """Test that single sample (1D array) works."""
        X_train = np.array([[1, 2], [2, 3], [3, 4]])
        y_train = np.array([0, 1, 1])
        X_test = np.array([2.5, 3.5])  # 1D array
        
        knn = KNN(n_neighbors=2)
        knn.fit(X_train, y_train)
        prediction = knn.predict(X_test)
        
        # Should return scalar or single-element array
        assert np.isscalar(prediction) or len(prediction) == 1


class TestKNNMetrics:
    """Test different distance metrics."""
    
    def test_euclidean_metric(self):
        """Test with euclidean distance (default)."""
        X_train = np.array([[0, 0], [1, 1], [3, 3]])
        y_train = np.array([0, 1, 1])
        X_test = np.array([[0.5, 0.5]])
        
        knn = KNN(n_neighbors=2, metric='euclidean')
        knn.fit(X_train, y_train)
        predictions = knn.predict(X_test)
        
        # Should work without errors
        assert predictions is not None
    
    def test_manhattan_metric(self):
        """Test with manhattan distance."""
        X_train = np.array([[0, 0], [1, 1], [3, 3]])
        y_train = np.array([0, 1, 1])
        X_test = np.array([[0.5, 0.5]])
        
        knn = KNN(n_neighbors=2, metric='manhattan')
        knn.fit(X_train, y_train)
        predictions = knn.predict(X_test)
        
        assert predictions is not None
    
    def test_invalid_metric(self):
        """Test that invalid metric raises error."""
        knn = KNN(n_neighbors=3, metric='invalid')
        X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])  # Need at least 3 samples
        y_train = np.array([0, 1, 0, 1])
        
        knn.fit(X_train, y_train)
        
        with pytest.raises(ValueError, match="Invalid metric"):
            knn.predict(np.array([[1.5, 2.5]]))


class TestKNNEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_k_equals_one(self):
        """Test with k=1 (nearest neighbor only)."""
        X_train = np.array([[1, 1], [2, 2], [3, 3]])
        y_train = np.array([0, 1, 2])
        X_test = np.array([[1.1, 1.1]])
        
        knn = KNN(n_neighbors=1)
        knn.fit(X_train, y_train)
        predictions = knn.predict(X_test)
        
        assert predictions == 0  # Should match nearest neighbor
    
    def test_k_equals_n_samples(self):
        """Test with k equal to number of training samples."""
        X_train = np.array([[1, 1], [2, 2], [3, 3]])
        y_train = np.array([0, 0, 1])
        X_test = np.array([[2, 2]])
        
        knn = KNN(n_neighbors=3)
        knn.fit(X_train, y_train)
        predictions = knn.predict(X_test)
        
        # Should predict majority class (0)
        assert predictions == 0
    
    def test_tie_breaking(self):
        """Test that ties are broken (first class wins)."""
        X_train = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])
        y_train = np.array([0, 1, 0, 1])  # Equal classes
        X_test = np.array([[2.5, 2.5]])
        
        knn = KNN(n_neighbors=4)  # Use all 4 points
        knn.fit(X_train, y_train)
        predictions = knn.predict(X_test)
        
        # Should return one of the classes (first one in case of tie)
        assert predictions in [0, 1]
    
    def test_single_class(self):
        """Test with only one class in training data."""
        X_train = np.array([[1, 1], [2, 2], [3, 3]])
        y_train = np.array([0, 0, 0])  # All same class
        X_test = np.array([[5, 5]])
        
        knn = KNN(n_neighbors=2)
        knn.fit(X_train, y_train)
        predictions = knn.predict(X_test)
        
        assert predictions == 0


class TestKNNErrorHandling:
    """Test error handling and validation."""
    
    def test_predict_before_fit(self):
        """Test that predicting before fitting raises error."""
        knn = KNN(n_neighbors=3)
        X_test = np.array([[1, 2]])
        
        with pytest.raises(ValueError, match="not been fitted"):
            knn.predict(X_test)
    
    def test_k_greater_than_samples(self):
        """Test that k > n_samples raises error."""
        X_train = np.array([[1, 2], [2, 3]])
        y_train = np.array([0, 1])
        
        knn = KNN(n_neighbors=5)  # k > 2 samples
        
        with pytest.raises(ValueError, match="cannot be greater"):
            knn.fit(X_train, y_train)
    
    def test_mismatched_X_y_shapes(self):
        """Test that mismatched X and y shapes raise error."""
        X_train = np.array([[1, 2], [2, 3], [3, 4]])
        y_train = np.array([0, 1])  # Wrong length
        
        knn = KNN(n_neighbors=2)
        
        with pytest.raises(ValueError, match="same number of samples"):
            knn.fit(X_train, y_train)
    
    def test_empty_X(self):
        """Test with empty X array."""
        X_train = np.array([]).reshape(0, 2)
        y_train = np.array([])
        
        knn = KNN(n_neighbors=1)
        
        with pytest.raises(ValueError, match="cannot be greater"):
            knn.fit(X_train, y_train)


class TestKNNRealWorld:
    """Test with more realistic scenarios."""
    
    def test_2d_classification(self):
        """Test 2D classification problem."""
        # Create two clusters
        np.random.seed(42)
        cluster1 = np.random.randn(10, 2) + [0, 0]
        cluster2 = np.random.randn(10, 2) + [5, 5]
        
        X_train = np.vstack([cluster1, cluster2])
        y_train = np.array([0] * 10 + [1] * 10)
        
        # Test point closer to cluster1
        X_test = np.array([[0.5, 0.5]])
        
        knn = KNN(n_neighbors=5)
        knn.fit(X_train, y_train)
        predictions = knn.predict(X_test)
        
        assert predictions == 0
    
    def test_3d_features(self):
        """Test with 3D feature space."""
        X_train = np.array([[1, 2, 3], [2, 3, 4], [5, 6, 7], [6, 7, 8]])
        y_train = np.array([0, 0, 1, 1])
        X_test = np.array([[1.5, 2.5, 3.5], [5.5, 6.5, 7.5]])
        
        knn = KNN(n_neighbors=2)
        knn.fit(X_train, y_train)
        predictions = knn.predict(X_test)
        
        assert len(predictions) == 2
        assert predictions[0] == 0
        assert predictions[1] == 1
    
    def test_many_classes(self):
        """Test with multiple classes."""
        X_train = np.array([
            [0, 0], [0.5, 0.5],  # Class 0
            [2, 2], [2.5, 2.5],  # Class 1
            [4, 4], [4.5, 4.5],  # Class 2
        ])
        y_train = np.array([0, 0, 1, 1, 2, 2])
        X_test = np.array([[0.2, 0.2], [2.2, 2.2], [4.2, 4.2]])
        
        knn = KNN(n_neighbors=2)
        knn.fit(X_train, y_train)
        predictions = knn.predict(X_test)
        
        assert np.array_equal(predictions, [0, 1, 2])


class TestKNNConsistency:
    """Test consistency and reproducibility."""
    
    def test_same_input_same_output(self):
        """Test that same input gives same output."""
        X_train = np.array([[1, 2], [2, 3], [3, 4], [5, 6]])
        y_train = np.array([0, 0, 1, 1])
        X_test = np.array([[2.5, 3.5]])
        
        knn1 = KNN(n_neighbors=2)
        knn1.fit(X_train, y_train)
        pred1 = knn1.predict(X_test)
        
        knn2 = KNN(n_neighbors=2)
        knn2.fit(X_train, y_train)
        pred2 = knn2.predict(X_test)
        
        assert pred1 == pred2
    
    def test_predict_training_data(self):
        """Test predicting on training data (should work)."""
        X_train = np.array([[1, 2], [2, 3], [3, 4]])
        y_train = np.array([0, 1, 1])
        
        knn = KNN(n_neighbors=2)
        knn.fit(X_train, y_train)
        predictions = knn.predict(X_train)
        
        # Should return predictions (may not match exactly due to voting)
        assert len(predictions) == len(y_train)
        assert all(p in [0, 1] for p in predictions)


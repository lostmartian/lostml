"""
Tests for Random Forest classifier and regressor.
"""
import numpy as np
import pytest
from lostml.tree.random_forest import RandomForest


class TestRandomForestBasic:
    """Basic functionality tests."""
    
    def test_fit_and_predict_classification(self):
        """Test basic fit and predict for classification."""
        X = np.array([[1, 2], [2, 3], [3, 4], [5, 6], [6, 7], [7, 8]])
        y = np.array([0, 0, 0, 1, 1, 1])
        
        rf = RandomForest(n_estimators=10, criterion='gini', random_state=42)
        rf.fit(X, y)
        predictions = rf.predict(X)
        
        # Should return predictions
        assert len(predictions) == len(y)
        assert all(p in [0, 1] for p in predictions)
    
    def test_fit_and_predict_regression(self):
        """Test basic fit and predict for regression."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
        y = np.array([2, 3, 4, 5, 6])
        
        rf = RandomForest(n_estimators=10, criterion='mse', random_state=42)
        rf.fit(X, y)
        predictions = rf.predict(X)
        
        # Should return numeric predictions
        assert len(predictions) == len(y)
        assert all(isinstance(p, (int, float, np.integer, np.floating)) for p in predictions)
    
    def test_predict_proba_classification(self):
        """Test predict_proba for classification."""
        X = np.array([[1, 2], [2, 3], [3, 4], [5, 6], [6, 7], [7, 8]])
        y = np.array([0, 0, 0, 1, 1, 1])
        
        rf = RandomForest(n_estimators=10, criterion='gini', random_state=42)
        rf.fit(X, y)
        probabilities = rf.predict_proba(X)
        
        # Should return probabilities
        assert probabilities.shape == (len(X), 2)  # 2 classes
        assert np.allclose(probabilities.sum(axis=1), 1.0)  # Probabilities sum to 1
        assert np.all(probabilities >= 0) and np.all(probabilities <= 1)
    
    def test_predict_proba_regression_error(self):
        """Test that predict_proba raises error for regression."""
        X = np.array([[1, 2], [2, 3], [3, 4]])
        y = np.array([2, 3, 4])
        
        rf = RandomForest(n_estimators=10, criterion='mse', random_state=42)
        rf.fit(X, y)
        
        with pytest.raises(ValueError, match="predict_proba is only available for classification"):
            rf.predict_proba(X)
    
    def test_single_tree(self):
        """Test with n_estimators=1 (essentially a decision tree with random features)."""
        X = np.array([[1, 2], [2, 3], [3, 4], [5, 6]])
        y = np.array([0, 0, 1, 1])
        
        rf = RandomForest(n_estimators=1, random_state=42)
        rf.fit(X, y)
        predictions = rf.predict(X)
        
        assert len(predictions) == len(y)
    
    def test_bootstrap_false(self):
        """Test with bootstrap=False (uses all data for each tree)."""
        X = np.array([[1, 2], [2, 3], [3, 4], [5, 6]])
        y = np.array([0, 0, 1, 1])
        
        rf = RandomForest(n_estimators=5, bootstrap=False, random_state=42)
        rf.fit(X, y)
        predictions = rf.predict(X)
        
        assert len(predictions) == len(y)
    
    def test_max_features_options(self):
        """Test different max_features options."""
        X = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [5, 6, 7]])
        y = np.array([0, 0, 1, 1])
        
        # Test 'sqrt'
        rf1 = RandomForest(n_estimators=5, max_features='sqrt', random_state=42)
        rf1.fit(X, y)
        assert len(rf1.predict(X)) == len(y)
        
        # Test 'log2'
        rf2 = RandomForest(n_estimators=5, max_features='log2', random_state=42)
        rf2.fit(X, y)
        assert len(rf2.predict(X)) == len(y)
        
        # Test int
        rf3 = RandomForest(n_estimators=5, max_features=2, random_state=42)
        rf3.fit(X, y)
        assert len(rf3.predict(X)) == len(y)
        
        # Test float
        rf4 = RandomForest(n_estimators=5, max_features=0.5, random_state=42)
        rf4.fit(X, y)
        assert len(rf4.predict(X)) == len(y)
        
        # Test None
        rf5 = RandomForest(n_estimators=5, max_features=None, random_state=42)
        rf5.fit(X, y)
        assert len(rf5.predict(X)) == len(y)


class TestRandomForestEdgeCases:
    """Edge case tests."""
    
    def test_single_sample(self):
        """Test with single training sample."""
        X = np.array([[1, 2]])
        y = np.array([0])
        
        rf = RandomForest(n_estimators=5, random_state=42)
        rf.fit(X, y)
        predictions = rf.predict(X)
        
        assert predictions[0] == 0
    
    def test_single_feature(self):
        """Test with single feature."""
        X = np.array([[1], [2], [3], [4]])
        y = np.array([0, 0, 1, 1])
        
        rf = RandomForest(n_estimators=5, random_state=42)
        rf.fit(X, y)
        predictions = rf.predict(X)
        
        assert len(predictions) == len(y)
    
    def test_empty_predict(self):
        """Test predicting on empty array."""
        X = np.array([[1, 2], [2, 3], [3, 4]])
        y = np.array([0, 0, 1])
        
        rf = RandomForest(n_estimators=5, random_state=42)
        rf.fit(X, y)
        
        X_empty = np.array([]).reshape(0, 2)
        predictions = rf.predict(X_empty)
        
        assert len(predictions) == 0
    
    def test_wrong_feature_count(self):
        """Test error when predicting with wrong number of features."""
        X = np.array([[1, 2], [2, 3], [3, 4]])
        y = np.array([0, 0, 1])
        
        rf = RandomForest(n_estimators=5, random_state=42)
        rf.fit(X, y)
        
        X_wrong = np.array([[1, 2, 3], [2, 3, 4]])
        
        with pytest.raises(ValueError, match="features"):
            rf.predict(X_wrong)
    
    def test_not_fitted_error(self):
        """Test error when predicting without fitting."""
        rf = RandomForest(n_estimators=5)
        X = np.array([[1, 2], [2, 3]])
        
        with pytest.raises(ValueError, match="not been fitted"):
            rf.predict(X)


class TestRandomForestReproducibility:
    """Tests for reproducibility with random_state."""
    
    def test_reproducibility_classification(self):
        """Test that same random_state produces same results."""
        X = np.array([[1, 2], [2, 3], [3, 4], [5, 6], [6, 7], [7, 8]])
        y = np.array([0, 0, 0, 1, 1, 1])
        
        rf1 = RandomForest(n_estimators=10, random_state=42)
        rf1.fit(X, y)
        predictions1 = rf1.predict(X)
        
        rf2 = RandomForest(n_estimators=10, random_state=42)
        rf2.fit(X, y)
        predictions2 = rf2.predict(X)
        
        # Should produce same predictions with same random_state
        np.testing.assert_array_equal(predictions1, predictions2)
    
    def test_different_random_state(self):
        """Test that different random_state can produce different results."""
        X = np.array([[1, 2], [2, 3], [3, 4], [5, 6], [6, 7], [7, 8]])
        y = np.array([0, 0, 0, 1, 1, 1])
        
        rf1 = RandomForest(n_estimators=10, random_state=42)
        rf1.fit(X, y)
        predictions1 = rf1.predict(X)
        
        rf2 = RandomForest(n_estimators=10, random_state=123)
        rf2.fit(X, y)
        predictions2 = rf2.predict(X)
        
        # May or may not be different, but should both be valid
        assert len(predictions1) == len(predictions2) == len(y)


class TestRandomForestPerformance:
    """Performance and behavior tests."""
    
    def test_ensemble_voting(self):
        """Test that ensemble makes reasonable predictions."""
        # Create a simple dataset
        np.random.seed(42)
        X = np.random.randn(100, 2)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        
        rf = RandomForest(n_estimators=50, random_state=42)
        rf.fit(X, y)
        predictions = rf.predict(X)
        
        # Should predict valid classes
        assert all(p in [0, 1] for p in predictions)
        
        # Accuracy should be reasonable (at least better than random)
        accuracy = np.mean(predictions == y)
        assert accuracy > 0.5  # Better than random guessing
    
    def test_regression_averaging(self):
        """Test that regression averages predictions correctly."""
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([2, 4, 6, 8, 10])  # Linear relationship
        
        rf = RandomForest(n_estimators=20, criterion='mse', random_state=42)
        rf.fit(X, y)
        predictions = rf.predict(X)
        
        # Predictions should be numeric
        assert all(isinstance(p, (int, float, np.integer, np.floating)) for p in predictions)
        
        # Should be close to actual values (allowing some error)
        errors = np.abs(predictions - y)
        assert np.mean(errors) < 3  # Reasonable error tolerance


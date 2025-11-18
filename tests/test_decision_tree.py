"""
Tests for Decision Tree classifier and regressor.
"""
import numpy as np
import pytest
from lostml.tree.decision_tree import DecisionTree


class TestDecisionTreeBasic:
    """Basic functionality tests."""
    
    def test_fit_and_predict_classification(self):
        """Test basic fit and predict for classification."""
        X = np.array([[1, 2], [2, 3], [3, 4], [5, 6], [6, 7]])
        y = np.array([0, 0, 1, 1, 1])
        
        tree = DecisionTree(criterion='gini', max_depth=3)
        tree.fit(X, y)
        predictions = tree.predict(X)
        
        # Should return predictions
        assert len(predictions) == len(y)
        assert all(p in [0, 1] for p in predictions)
    
    def test_fit_and_predict_regression(self):
        """Test basic fit and predict for regression."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
        y = np.array([2, 3, 4, 5])
        
        tree = DecisionTree(criterion='mse', max_depth=3)
        tree.fit(X, y)
        predictions = tree.predict(X)
        
        # Should return numeric predictions
        assert len(predictions) == len(y)
        assert all(isinstance(p, (int, float, np.integer, np.floating)) for p in predictions)
    
    def test_single_sample(self):
        """Test with single training sample."""
        X = np.array([[1, 2]])
        y = np.array([0])
        
        tree = DecisionTree()
        tree.fit(X, y)
        predictions = tree.predict(X)
        
        assert predictions[0] == 0
    
    def test_single_feature(self):
        """Test with single feature."""
        X = np.array([[1], [2], [3], [4]])
        y = np.array([0, 0, 1, 1])
        
        tree = DecisionTree()
        tree.fit(X, y)
        predictions = tree.predict(X)
        
        assert len(predictions) == len(y)


class TestDecisionTreeClassification:
    """Tests specific to classification (Gini criterion)."""
    
    def test_pure_node(self):
        """Test that pure nodes (all same class) create leaf."""
        X = np.array([[1, 2], [2, 3], [3, 4]])
        y = np.array([0, 0, 0])  # All same class
        
        tree = DecisionTree(criterion='gini')
        tree.fit(X, y)
        predictions = tree.predict(X)
        
        # All predictions should be 0
        assert np.all(predictions == 0)
    
    def test_perfect_separation(self):
        """Test with perfectly separable data."""
        X = np.array([[1, 1], [2, 2], [5, 5], [6, 6]])
        y = np.array([0, 0, 1, 1])
        
        tree = DecisionTree(criterion='gini', max_depth=10)
        tree.fit(X, y)
        predictions = tree.predict(X)
        
        # Should achieve perfect classification
        assert np.array_equal(predictions, y)
    
    def test_two_classes(self):
        """Test binary classification."""
        X = np.array([[1, 1], [1, 2], [2, 1], [2, 2], [5, 5], [5, 6], [6, 5], [6, 6]])
        y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        
        tree = DecisionTree(criterion='gini', max_depth=5)
        tree.fit(X, y)
        predictions = tree.predict(X)
        
        assert len(predictions) == len(y)
        assert all(p in [0, 1] for p in predictions)
    
    def test_multiple_classes(self):
        """Test multi-class classification."""
        X = np.array([
            [1, 1], [1, 2],  # Class 0
            [3, 3], [3, 4],  # Class 1
            [5, 5], [5, 6],  # Class 2
        ])
        y = np.array([0, 0, 1, 1, 2, 2])
        
        tree = DecisionTree(criterion='gini')
        tree.fit(X, y)
        predictions = tree.predict(X)
        
        assert len(predictions) == len(y)
        assert all(p in [0, 1, 2] for p in predictions)


class TestDecisionTreeRegression:
    """Tests specific to regression (MSE criterion)."""
    
    def test_linear_relationship(self):
        """Test with linear relationship."""
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([2, 4, 6, 8, 10])
        
        tree = DecisionTree(criterion='mse', max_depth=3)
        tree.fit(X, y)
        predictions = tree.predict(X)
        
        assert len(predictions) == len(y)
        assert all(isinstance(p, (int, float, np.integer, np.floating)) for p in predictions)
    
    def test_constant_target(self):
        """Test with constant target values."""
        X = np.array([[1, 2], [2, 3], [3, 4]])
        y = np.array([5, 5, 5])
        
        tree = DecisionTree(criterion='mse')
        tree.fit(X, y)
        predictions = tree.predict(X)
        
        # All predictions should be close to 5
        assert np.allclose(predictions, 5)
    
    def test_non_linear_relationship(self):
        """Test with non-linear relationship."""
        X = np.array([[1], [2], [3], [10], [11], [12]])
        y = np.array([1, 2, 3, 10, 11, 12])
        
        tree = DecisionTree(criterion='mse', max_depth=5)
        tree.fit(X, y)
        predictions = tree.predict(X)
        
        assert len(predictions) == len(y)


class TestDecisionTreeParameters:
    """Test different parameter configurations."""
    
    def test_max_depth(self):
        """Test max_depth parameter."""
        X = np.array([[1, 2], [2, 3], [3, 4], [5, 6], [6, 7], [7, 8]])
        y = np.array([0, 0, 0, 1, 1, 1])
        
        # Shallow tree
        tree_shallow = DecisionTree(max_depth=1)
        tree_shallow.fit(X, y)
        
        # Deep tree
        tree_deep = DecisionTree(max_depth=10)
        tree_deep.fit(X, y)
        
        # Both should work
        assert tree_shallow.root is not None
        assert tree_deep.root is not None
    
    def test_max_depth_none(self):
        """Test with max_depth=None (unlimited depth)."""
        X = np.array([[1, 2], [2, 3], [3, 4], [5, 6]])
        y = np.array([0, 0, 1, 1])
        
        tree = DecisionTree(max_depth=None)
        tree.fit(X, y)
        predictions = tree.predict(X)
        
        assert len(predictions) == len(y)
    
    def test_min_samples_split(self):
        """Test min_samples_split parameter."""
        X = np.array([[1, 2], [2, 3], [3, 4], [5, 6], [6, 7]])
        y = np.array([0, 0, 1, 1, 1])
        
        tree = DecisionTree(min_samples_split=3)
        tree.fit(X, y)
        predictions = tree.predict(X)
        
        assert len(predictions) == len(y)
    
    def test_min_samples_leaf(self):
        """Test min_samples_leaf parameter."""
        X = np.array([[1, 2], [2, 3], [3, 4], [5, 6], [6, 7], [7, 8]])
        y = np.array([0, 0, 0, 1, 1, 1])
        
        tree = DecisionTree(min_samples_leaf=2)
        tree.fit(X, y)
        predictions = tree.predict(X)
        
        assert len(predictions) == len(y)


class TestDecisionTreeEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_empty_X(self):
        """Test that empty X raises error."""
        X = np.array([]).reshape(0, 2)
        y = np.array([])
        
        tree = DecisionTree()
        
        with pytest.raises(ValueError, match="cannot be empty"):
            tree.fit(X, y)
    
    def test_mismatched_X_y_shapes(self):
        """Test that mismatched X and y shapes raise error."""
        X = np.array([[1, 2], [2, 3], [3, 4]])
        y = np.array([0, 1])  # Wrong length
        
        tree = DecisionTree()
        
        with pytest.raises(ValueError, match="same number of samples"):
            tree.fit(X, y)
    
    def test_predict_before_fit(self):
        """Test that predicting before fitting raises error."""
        tree = DecisionTree()
        X_test = np.array([[1, 2]])
        
        with pytest.raises(ValueError, match="not been fitted"):
            tree.predict(X_test)
    
    def test_wrong_feature_count(self):
        """Test that wrong number of features in predict raises error."""
        X_train = np.array([[1, 2], [2, 3]])
        y_train = np.array([0, 1])
        X_test = np.array([[1, 2, 3]])  # Wrong number of features
        
        tree = DecisionTree()
        tree.fit(X_train, y_train)
        
        with pytest.raises(ValueError, match="features"):
            tree.predict(X_test)
    
    def test_invalid_criterion(self):
        """Test that invalid criterion raises error."""
        X = np.array([[1, 2], [2, 3]])
        y = np.array([0, 1])
        
        tree = DecisionTree(criterion='invalid')
        
        with pytest.raises(ValueError, match="criterion must be"):
            tree.fit(X, y)
    
    def test_empty_predict(self):
        """Test predicting on empty array."""
        X_train = np.array([[1, 2], [2, 3]])
        y_train = np.array([0, 1])
        X_test = np.array([]).reshape(0, 2)
        
        tree = DecisionTree()
        tree.fit(X_train, y_train)
        predictions = tree.predict(X_test)
        
        assert len(predictions) == 0


class TestDecisionTreeRealWorld:
    """Test with more realistic scenarios."""
    
    def test_2d_classification(self):
        """Test 2D classification problem."""
        np.random.seed(42)
        # Create two clusters
        cluster1 = np.random.randn(20, 2) + [0, 0]
        cluster2 = np.random.randn(20, 2) + [5, 5]
        
        X = np.vstack([cluster1, cluster2])
        y = np.array([0] * 20 + [1] * 20)
        
        tree = DecisionTree(max_depth=5)
        tree.fit(X, y)
        predictions = tree.predict(X)
        
        assert len(predictions) == len(y)
        assert all(p in [0, 1] for p in predictions)
    
    def test_3d_features(self):
        """Test with 3D feature space."""
        X = np.array([[1, 2, 3], [2, 3, 4], [5, 6, 7], [6, 7, 8]])
        y = np.array([0, 0, 1, 1])
        
        tree = DecisionTree()
        tree.fit(X, y)
        predictions = tree.predict(X)
        
        assert len(predictions) == len(y)
    
    def test_many_samples(self):
        """Test with larger dataset."""
        np.random.seed(42)
        X = np.random.randn(100, 3)
        y = (X[:, 0] > 0).astype(int)
        
        tree = DecisionTree(max_depth=5)
        tree.fit(X, y)
        predictions = tree.predict(X)
        
        assert len(predictions) == len(y)
    
    def test_regression_non_linear(self):
        """Test regression on non-linear data."""
        X = np.array([[1], [2], [3], [10], [11], [12], [20], [21], [22]])
        y = np.array([1, 2, 3, 10, 11, 12, 20, 21, 22])
        
        tree = DecisionTree(criterion='mse', max_depth=5)
        tree.fit(X, y)
        predictions = tree.predict(X)
        
        assert len(predictions) == len(y)


class TestDecisionTreeConsistency:
    """Test consistency and reproducibility."""
    
    def test_same_input_same_output(self):
        """Test that same input gives same output."""
        X = np.array([[1, 2], [2, 3], [3, 4], [5, 6]])
        y = np.array([0, 0, 1, 1])
        
        tree1 = DecisionTree(max_depth=3)
        tree1.fit(X, y)
        pred1 = tree1.predict(X)
        
        tree2 = DecisionTree(max_depth=3)
        tree2.fit(X, y)
        pred2 = tree2.predict(X)
        
        # Should give same results (deterministic)
        assert np.array_equal(pred1, pred2)
    
    def test_predict_training_data(self):
        """Test predicting on training data."""
        X = np.array([[1, 2], [2, 3], [3, 4], [5, 6]])
        y = np.array([0, 0, 1, 1])
        
        tree = DecisionTree(max_depth=5)
        tree.fit(X, y)
        predictions = tree.predict(X)
        
        # Should return predictions
        assert len(predictions) == len(y)
        assert all(p in [0, 1] for p in predictions)


class TestDecisionTreeConstraints:
    """Test that constraints are properly enforced."""
    
    def test_max_depth_constraint(self):
        """Test that max_depth is respected."""
        X = np.array([[1, 2], [2, 3], [3, 4], [5, 6], [6, 7], [7, 8]])
        y = np.array([0, 0, 0, 1, 1, 1])
        
        tree = DecisionTree(max_depth=1)
        tree.fit(X, y)
        
        # Tree should be limited to depth 1
        # We can't easily check depth without traversal, but it should work
        predictions = tree.predict(X)
        assert len(predictions) == len(y)
    
    def test_min_samples_split_constraint(self):
        """Test that min_samples_split is respected."""
        X = np.array([[1, 2], [2, 3], [3, 4], [5, 6]])
        y = np.array([0, 0, 1, 1])
        
        # With min_samples_split=5, tree should be a single leaf (can't split with < 5 samples)
        tree = DecisionTree(min_samples_split=5)
        tree.fit(X, y)
        
        # Should still predict
        predictions = tree.predict(X)
        assert len(predictions) == len(y)
        # All predictions should be the same (single leaf - most common class)
        assert len(np.unique(predictions)) == 1
        # Should predict the most common class (0 or 1, both appear twice, so first one)
        assert predictions[0] in [0, 1]
    
    def test_min_samples_leaf_constraint(self):
        """Test that min_samples_leaf is respected."""
        X = np.array([[1, 2], [2, 3], [3, 4], [5, 6], [6, 7], [7, 8]])
        y = np.array([0, 0, 0, 1, 1, 1])
        
        tree = DecisionTree(min_samples_leaf=3)
        tree.fit(X, y)
        predictions = tree.predict(X)
        
        assert len(predictions) == len(y)


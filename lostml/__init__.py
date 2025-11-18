"""
lostml - A from-scratch machine learning library.

This package provides implementations of various machine learning algorithms
from scratch, including linear models, tree-based models, and more.
"""

from .linear_models import (
    LinearRegression,
    RigdeRegression,
    LassoRegression,
    ElasticNet,
    LogisticRegression,
)

__all__ = [
    "LinearRegression",
    "RigdeRegression",
    "LassoRegression",
    "ElasticNet",
    "LogisticRegression",
]

__version__ = "0.1.0"


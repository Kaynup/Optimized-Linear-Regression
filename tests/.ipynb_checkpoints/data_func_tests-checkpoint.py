import pytest
import numpy as np
from OptLinearRegress.utils.data import standardize, shuffle_arrays, train_test_split

def test_standardize():
    X = np.array([[1, 2], [3, 4]])
    X_scaled, mean, std = standardize(X)
    assert np.allclose(X_scaled.mean(axis=0), 0)
    assert np.allclose(X_scaled.std(axis=0), 1)

def test_shuffle_arrays():
    X = np.array([[1], [2], [3]])
    y = np.array([1, 2, 3])
    X_sh, y_sh = shuffle_arrays(X, y, seed=42)
    assert set(X_sh.flatten()) == {1, 2, 3}
    assert set(y_sh) == {1, 2, 3}

def test_train_test_split():
    X = np.arange(10).reshape(10, 1)
    y = np.arange(10)
    X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.2, seed=42)
    assert len(X_train) == 8
    assert len(X_test) == 2

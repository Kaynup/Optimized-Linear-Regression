import pytest
from OptLinearRegress.utils.metrics import mean_squared_error, mean_absolute_error, r2_score

def test_mse():
    y_true = [1, 2, 3]
    y_pred = [1, 2, 3]
    assert mean_squared_error(y_true, y_pred) == 0

def test_mae():
    y_true = [1, 2, 3]
    y_pred = [2, 2, 2]
    assert mean_absolute_error(y_true, y_pred) == pytest.approx(0.6666666, 1e-6)

def test_r2():
    y_true = [1, 2, 3]
    y_pred = [1, 2, 3]
    assert r2_score(y_true, y_pred) == 1.0

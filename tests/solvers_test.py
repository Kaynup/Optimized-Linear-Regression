import pytest
from OptLinearRegress.models.linear_model import LinearRegressor
import numpy as np

def test_linear_fit_predict():
    X = np.array([[1], [2], [3], [4]])
    y = np.array([2, 4, 6, 8])
    model = LinearRegressor()
    coef = model.fit(X.tolist(), y.tolist())
    y_pred = model.predict(X.tolist())
    assert all(abs(y_pred[i]-y[i]) < 1e-6 for i in range(len(y)))

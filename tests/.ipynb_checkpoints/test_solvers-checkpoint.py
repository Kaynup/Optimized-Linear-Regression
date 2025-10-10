import numpy as np
from OptLinearRegress.solvers.normal_equation import solve_normal_equation

def test_solve_normal_equation():
    X = np.array([[1,2],[3,4],[5,6]], dtype=np.float64)
    y = np.array([1,2,3], dtype=np.float64)

    n_samples, n_features = X.shape[0], X.shape[1]+1
    X_with_intercept = np.ones((n_samples, n_features), dtype=np.float64)
    X_with_intercept[:,1:] = X
    beta = np.zeros(n_features, dtype=np.float64)

    err = solve_normal_equation(X_with_intercept.flatten(), y.flatten(), n_samples, n_features, 1e-8, beta)
    assert err == 0

    # predicted y
    y_pred = X_with_intercept @ beta
    assert np.allclose(y_pred, y, atol=1e-6)

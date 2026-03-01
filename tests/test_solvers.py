import array
import math
from OptLinearRegress.solvers.normal_equation import solve_normal_equation

def test_solve_normal_equation():
    # X_with_intercept flattened
    X_flat = array.array('d', [
        1.0, 1.0, 2.0,
        1.0, 3.0, 4.0,
        1.0, 5.0, 6.0
    ])
    y = array.array('d', [1.0, 2.0, 3.0])
    
    n_samples, n_features = 3, 3
    beta = array.array('d', [0.0, 0.0, 0.0])
    
    err = solve_normal_equation(X_flat, y, n_samples, n_features, 1e-8, beta)
    assert err == 0
    
    # predicted y = X * beta
    y_pred = []
    for i in range(n_samples):
        val = sum(X_flat[i * n_features + j] * beta[j] for j in range(n_features))
        y_pred.append(val)
        
    for py, ty in zip(y_pred, y):
        assert math.isclose(py, ty, abs_tol=1e-6)

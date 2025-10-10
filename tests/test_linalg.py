import numpy as np
from OptLinearRegress.linalg import py_matmul, py_matvec, py_transpose, py_invert

def test_matmul():
    A = np.array([[1,2,3],[4,5,6]], dtype=np.float64).flatten()
    B = np.array([[7,8],[9,10],[11,12]], dtype=np.float64).flatten()
    C = np.zeros(4, dtype=np.float64)
    py_matmul(A, B, C, 2, 3, 2)
    assert np.allclose(C, [58,64,139,154])

def test_matvec():
    A = np.array([[1,2,3],[4,5,6]], dtype=np.float64).flatten()
    x = np.array([7,8,9], dtype=np.float64)
    y = np.zeros(2, dtype=np.float64)
    py_matvec(A, x, y, 2, 3)
    assert np.allclose(y, [50, 122])

def test_transpose():
    A = np.array([[1,2],[3,4]], dtype=np.float64).flatten()
    py_transpose(A, 2)
    assert np.allclose(A, [1,3,2,4])

def test_invert():
    A = np.array([[4,7],[2,6]], dtype=np.float64).flatten()
    err = py_invert(A, 2)
    assert err == 0
    det = 4*6-2*7
    A_inv_expected = np.array([6,-7,-2,4], dtype=np.float64)/det
    assert np.allclose(A, A_inv_expected)

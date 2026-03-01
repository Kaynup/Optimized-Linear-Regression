import array
import math
from OptLinearRegress.linalg.linalg import py_matmul, py_matvec, py_transpose, py_invert

def allclose(a, b, atol=1e-6):
    if len(a) != len(b): return False
    return all(math.isclose(x, y, abs_tol=atol) for x, y in zip(a, b))

def test_matmul():
    A = array.array('d', [1,2,3,4,5,6])
    B = array.array('d', [7,8,9,10,11,12])
    C = array.array('d', [0,0,0,0])
    py_matmul(A, B, C, 2, 3, 2)
    assert allclose(C, [58,64,139,154])

def test_matvec():
    A = array.array('d', [1,2,3,4,5,6])
    x = array.array('d', [7,8,9])
    y = array.array('d', [0,0])
    py_matvec(A, x, y, 2, 3)
    assert allclose(y, [50, 122])

def test_transpose():
    A = array.array('d', [1,2,3,4])
    py_transpose(A, 2)
    assert allclose(A, [1,3,2,4])

def test_invert():
    A = array.array('d', [4,7,2,6])
    err = py_invert(A, 2)
    assert err == 0
    det = 4*6-2*7
    A_inv_expected = [6/det, -7/det, -2/det, 4/det]
    assert allclose(A, A_inv_expected)

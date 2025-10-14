# linalg/__init__.py
"""
Lightweight Cython linear algebra library.
Pure C-memory operations, NumPy-free, debuggable.
"""

from array import array
from .linalg import matmul, matvec, transpose, invert

__all__ = ["matmul", "matvec", "transpose", "invert"]


if __name__ == "__main__":
    print("Running quick linalg debug test...")

    # Prepare test data
    A = array("d", [1, 2, 3, 4])       # 2x2
    B = array("d", [5, 6, 7, 8])       # 2x2
    C = array("d", [0, 0, 0, 0])       # output buffer

    mvA = memoryview(A).cast("d", (2, 2))
    mvB = memoryview(B).cast("d", (2, 2))
    mvC = memoryview(C).cast("d", (2, 2))

    # Run tests
    matmul(mvA, mvB, mvC)
    print("A * B =", list(C))

    mvX = memoryview(array("d", [1, 1])).cast("d")
    mvY = memoryview(array("d", [0, 0])).cast("d")
    matvec(mvA, mvX, mvY)
    print("A * x =", list(mvY))

    transpose(mvA)
    print("Transpose(A) =", list(A))

    print("Invertible?", invert(mvB) == 0)

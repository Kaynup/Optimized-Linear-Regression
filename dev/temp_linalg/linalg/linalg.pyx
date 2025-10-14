# linalg/linalg.pyx
# cython: language_level=3
# cython: boundscheck=False, wraparound=False, cdivision=True
# distutils: language = c

from libc.stdlib cimport malloc, free
from libc.math cimport fabs
import cython


# ==============================================================
#  Low-level C implementations
# ==============================================================

cdef void transpose_inplace(double* A, int n):
    cdef int i, j
    cdef double tmp
    for i in range(n):
        for j in range(i + 1, n):
            tmp = A[i * n + j]
            A[i * n + j] = A[j * n + i]
            A[j * n + i] = tmp


cdef void matmul_c(double* A, double* B, double* C, int M, int N, int K):
    cdef int i, j, k
    cdef double s
    for i in range(M):
        for j in range(K):
            s = 0.0
            for k in range(N):
                s += A[i * N + k] * B[k * K + j]
            C[i * K + j] = s


cdef void matvec_c(double* A, double* x, double* y, int M, int N):
    cdef int i, j
    cdef double s
    for i in range(M):
        s = 0.0
        for j in range(N):
            s += A[i * N + j] * x[j]
        y[i] = s


cdef int invert_matrix(double* A, int n):
    """Simple LU-like elimination (checks invertibility)."""
    cdef int i, j, k, maxRow
    cdef double tmp, maxEl
    cdef int* ipiv = <int*> malloc(n * sizeof(int))
    if ipiv == NULL:
        return -1

    for i in range(n):
        ipiv[i] = i

    for i in range(n):
        maxEl = fabs(A[i * n + i])
        maxRow = i
        for k in range(i + 1, n):
            if fabs(A[k * n + i]) > maxEl:
                maxEl = fabs(A[k * n + i])
                maxRow = k

        if maxRow != i:
            for k in range(n):
                tmp = A[i * n + k]
                A[i * n + k] = A[maxRow * n + k]
                A[maxRow * n + k] = tmp

        if A[i * n + i] == 0.0:
            free(ipiv)
            return -1

        for k in range(i + 1, n):
            A[k * n + i] /= A[i * n + i]
            for j in range(i + 1, n):
                A[k * n + j] -= A[k * n + i] * A[i * n + j]

    free(ipiv)
    return 0


# ==============================================================
#  Public Cython functions (cpdef = callable from Python + C)
# ==============================================================

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void transpose(double[:, :] A):
    """In-place transpose of square matrix."""
    cdef int n = A.shape[0]
    if n != A.shape[1]:
        raise ValueError("Matrix must be square")
    transpose_inplace(&A[0, 0], n)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void matmul(double[:, :] A, double[:, :] B, double[:, :] C):
    """Direct matrix multiplication (no Python overhead)."""
    cdef int M = A.shape[0]
    cdef int N = A.shape[1]
    cdef int K = B.shape[1]
    if B.shape[0] != N or C.shape[0] != M or C.shape[1] != K:
        raise ValueError("Incompatible dimensions")
    matmul_c(&A[0, 0], &B[0, 0], &C[0, 0], M, N, K)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void matvec(double[:, :] A, double[:] x, double[:] y):
    """Direct matrix-vector multiplication."""
    cdef int M = A.shape[0]
    cdef int N = A.shape[1]
    if x.shape[0] != N or y.shape[0] != M:
        raise ValueError("Dimension mismatch")
    matvec_c(&A[0, 0], &x[0], &y[0], M, N)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef int invert(double[:, :] A):
    """In-place invertibility check (returns 0 if invertible, -1 otherwise)."""
    cdef int n = A.shape[0]
    if n != A.shape[1]:
        raise ValueError("Matrix must be square")
    return invert_matrix(&A[0, 0], n)

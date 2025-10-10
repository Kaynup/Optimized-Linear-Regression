# cython: language_level=3
# cython: boundscheck=False, wraparound=False, cdivision=True
# distutils: language = c++

import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free

# -----------------------------
# C-level functions
# -----------------------------
cdef void transpose_inplace(double* A, int n):
    cdef int i, j
    cdef double tmp
    for i in range(n):
        for j in range(i+1, n):
            tmp = A[i*n + j]
            A[i*n + j] = A[j*n + i]
            A[j*n + i] = tmp

cdef void matmul_c(double* A, double* B, double* C, int M, int N, int K):
    cdef int i, j, k
    cdef double s
    for i in range(M):
        for j in range(K):
            s = 0
            for k in range(N):
                s += A[i*N + k] * B[k*K + j]
            C[i*K + j] = s

cdef void matvec_c(double* A, double* x, double* y, int M, int N):
    cdef int i, j
    cdef double s
    for i in range(M):
        s = 0
        for j in range(N):
            s += A[i*N + j] * x[j]
        y[i] = s

cdef int invert_matrix(double* A, int n):
    cdef int i, j, k, maxRow
    cdef double tmp, maxEl
    cdef int* ipiv = <int*> malloc(n * sizeof(int))
    if ipiv == NULL:
        return -1
    for i in range(n):
        ipiv[i] = i
    for i in range(n):
        maxEl = abs(A[i*n + i])
        maxRow = i
        for k in range(i+1, n):
            if abs(A[k*n + i]) > maxEl:
                maxEl = abs(A[k*n + i])
                maxRow = k
        if maxRow != i:
            for k in range(n):
                tmp = A[i*n + k]
                A[i*n + k] = A[maxRow*n + k]
                A[maxRow*n + k] = tmp
        if A[i*n + i] == 0:
            free(ipiv)
            return -1
        for k in range(i+1, n):
            A[k*n + i] /= A[i*n + i]
            for j in range(i+1, n):
                A[k*n + j] -= A[k*n + i]*A[i*n + j]
    free(ipiv)
    return 0

# -----------------------------
# Python-callable wrappers
# -----------------------------
cpdef np.ndarray py_matmul(np.ndarray[double, ndim=1] A, np.ndarray[double, ndim=1] B, int M, int N, int K):
    """
    Matrix multiplication wrapper.
    A: flattened M x N
    B: flattened N x K
    Returns M x K numpy array
    """
    cdef double* C_ptr
    cdef np.ndarray[double, ndim=1] C = np.zeros(M*K, dtype=np.float64)
    C_ptr = <double*> C.ctypes.data
    matmul_c(&A[0], &B[0], C_ptr, M, N, K)
    return C.reshape((M,K))

cpdef np.ndarray py_matvec(np.ndarray[double, ndim=1] A, np.ndarray[double, ndim=1] x, int M, int N):
    """
    Matrix-vector multiplication wrapper.
    A: flattened M x N
    x: N
    Returns vector of length M
    """
    cdef np.ndarray[double, ndim=1] y = np.zeros(M, dtype=np.float64)
    matvec_c(&A[0], &x[0], &y[0], M, N)
    return y

cpdef np.ndarray py_transpose(np.ndarray[double, ndim=1] A, int n):
    """
    In-place transpose of n x n matrix (flattened).
    Returns reshaped numpy array.
    """
    transpose_inplace(&A[0], n)
    return A.reshape((n,n))

cpdef int py_invert(np.ndarray[double, ndim=1] A, int n):
    """
    In-place inversion of n x n matrix (flattened).
    Returns 0 on success, -1 if singular.
    """
    return invert_matrix(&A[0], n)

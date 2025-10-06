# cython: boundscheck=False, wraparound=False
cimport cython
from OptLinearRegress.linalg.linalg cimport *

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

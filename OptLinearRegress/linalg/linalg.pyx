# cython: language_level=3
# cython: boundscheck=False, wraparound=False, cdivision=True
# distutils: language = c++

from libc.stdlib cimport malloc, free

cdef void matmul_c(double* A, double* B, double* C, int m, int n, int p):
    cdef int i, j, k
    for i in range(m):
        for j in range(p):
            C[i*p + j] = 0.0
            for k in range(n):
                C[i*p + j] += A[i*n + k] * B[k*p + j]

cdef void matvec_c(double* A, double* x, double* y, int m, int n):
    cdef int i, j
    for i in range(m):
        y[i] = 0.0
        for j in range(n):
            y[i] += A[i*n + j] * x[j]

cdef void transpose_inplace(double* A, int n):
    cdef int i, j
    cdef double temp
    for i in range(n):
        for j in range(i+1, n):
            temp = A[i*n + j]
            A[i*n + j] = A[j*n + i]
            A[j*n + i] = temp

cdef int invert_matrix(double* A, int n):
    cdef int i, j, k
    cdef double pivot, temp
    
    cdef double* I = <double*> malloc(n * n * sizeof(double))
    if I == NULL:
        return -2
        
    for i in range(n):
        for j in range(n):
            I[i*n + j] = 1.0 if i == j else 0.0
            
    for i in range(n):
        pivot = A[i*n + i]
        if pivot == 0.0:
            free(I)
            return -1 # singular
            
        for j in range(n):
            A[i*n + j] /= pivot
            I[i*n + j] /= pivot
            
        for k in range(n):
            if k != i:
                temp = A[k*n + i]
                for j in range(n):
                    A[k*n + j] -= temp * A[i*n + j]
                    I[k*n + j] -= temp * I[i*n + j]
                    
    for i in range(n):
        for j in range(n):
            A[i*n + j] = I[i*n + j]
            
    free(I)
    return 0

# Python callable wrappers
cpdef py_matmul(double[:] A_flat, double[:] B_flat, double[:] C_flat, int m, int n, int p):
    matmul_c(&A_flat[0], &B_flat[0], &C_flat[0], m, n, p)

cpdef py_matvec(double[:] A_flat, double[:] x_flat, double[:] y_flat, int m, int n):
    matvec_c(&A_flat[0], &x_flat[0], &y_flat[0], m, n)

cpdef py_transpose(double[:] A_flat, int n):
    transpose_inplace(&A_flat[0], n)

cpdef int py_invert(double[:] A_flat, int n):
    return invert_matrix(&A_flat[0], n)

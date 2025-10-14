# linalg/linalg.pxd
# Low-level Cython declarations for internal and external C access

from libc.stdlib cimport malloc, free
from libc.math cimport fabs

# ---- Core C-level functions ----
cdef void transpose_inplace(double* A, int n)
cdef void matmul_c(double* A, double* B, double* C, int M, int N, int K)
cdef void matvec_c(double* A, double* x, double* y, int M, int N)
cdef int invert_matrix(double* A, int n)

# ---- Cython-level callable functions ----
cpdef void transpose(double[:, :] A)
cpdef void matmul(double[:, :] A, double[:, :] B, double[:, :] C)
cpdef void matvec(double[:, :] A, double[:] x, double[:] y)
cpdef int invert(double[:, :] A)

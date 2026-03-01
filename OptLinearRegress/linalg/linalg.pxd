# cython: language_level=3

cdef void matmul_c(double* A, double* B, double* C, int m, int n, int p)
cdef void matvec_c(double* A, double* x, double* y, int m, int n)
cdef void transpose_inplace(double* A, int n)
cdef int invert_matrix(double* A, int n)

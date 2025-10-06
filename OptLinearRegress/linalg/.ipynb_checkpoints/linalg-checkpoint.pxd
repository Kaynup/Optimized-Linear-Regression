# cython: language_level=3
# cython: boundscheck=False, wraparound=False, cdivision=True

cdef void matmul_c(double* A, double* B, double* C, int M, int N, int K)
cdef void matvec_c(double* A, double* x, double* y, int M, int N)
cdef void transpose_inplace(double* A, int n)
cdef int invert_matrix(double* A, int n)

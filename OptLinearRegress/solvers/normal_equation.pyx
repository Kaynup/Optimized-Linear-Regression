# cython: language_level=3
# cython: boundscheck=False, wraparound=False, cdivision=True
# distutils: language = c++

import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free
from OptLinearRegress.linalg.linalg cimport matmul_c, transpose_inplace, invert_matrix

# -----------------------------
# C-level solver (fast)
# -----------------------------
cdef int _solve_normal_equation(double* X, double* y, int n_samples, int n_features, double alpha, double* beta):
    """
    Solves (X^T X + alpha*I) beta = X^T y
    Returns 0 on success, -1 if singular, -2 if allocation fails
    """
    cdef int i, j, k, err
    cdef double* XtX = <double*> malloc(n_features * n_features * sizeof(double))
    cdef double* Xty = <double*> malloc(n_features * sizeof(double))

    if XtX == NULL or Xty == NULL:
        if XtX != NULL: free(XtX)
        if Xty != NULL: free(Xty)
        return -2

    # Compute XtX = X^T X + alpha*I
    for i in range(n_features):
        for j in range(n_features):
            XtX[i*n_features + j] = 0.0
            for k in range(n_samples):
                XtX[i*n_features + j] += X[k*n_features + i] * X[k*n_features + j]
        XtX[i*n_features + i] += alpha

    # Compute Xty = X^T y
    for i in range(n_features):
        Xty[i] = 0.0
        for k in range(n_samples):
            Xty[i] += X[k*n_features + i] * y[k]

    # Solve linear system
    err = invert_matrix(XtX, n_features)
    if err != 0:
        free(XtX)
        free(Xty)
        return -1

    # beta = XtX^-1 Xty
    for i in range(n_features):
        beta[i] = 0.0
        for j in range(n_features):
            beta[i] += XtX[i*n_features + j] * Xty[j]

    free(XtX)
    free(Xty)
    return 0

# -----------------------------
# Python-callable wrapper
# -----------------------------
cpdef solve_normal_equation(np.ndarray[double, ndim=1] X_flat,
                            np.ndarray[double, ndim=1] y,
                            int n_samples,
                            int n_features,
                            double alpha,
                            np.ndarray[double, ndim=1] beta):
    """
    Python wrapper for _solve_normal_equation.
    Expects flattened X (1D numpy array) and y/beta as 1D numpy arrays.
    """
    cdef double* X_ptr = <double*> X_flat.ctypes.data
    cdef double* y_ptr = <double*> y.ctypes.data
    cdef double* beta_ptr = <double*> beta.ctypes.data

    return _solve_normal_equation(X_ptr, y_ptr, n_samples, n_features, alpha, beta_ptr)

# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
# distutils: language = c++

from libc.math cimport fabs
from libc.stdlib cimport malloc, free
from cython cimport cclass, cfunc

# Transpose of matrix
cdef void transpose_inplace(double* MAT, double* MAT_t, int rows, int cols) nogil:
    cdef int i, j
    for i in range(rows):
        for j in range(cols):
            MAT_t[j * rows + i] = MAT[i * cols + j]

# Multiplication of two matrices
cdef void matmul_c(double* A, double* B, double* R, int A_rows, int A_cols, int B_cols) nogil:
    cdef int i, j, k
    cdef double temp
    for i in range(A_rows * B_cols):
        R[i] = 0
    for i in range(A_rows):
        for j in range(B_cols):
            temp = 0
            for k in range(A_cols):
                temp += A[i * A_cols + k] * B[k * B_cols + j]
            R[i * B_cols + j] = temp

# Multiplication of a vector and a matrix
cdef void matvec_c(double* M, double* v, double* r, int rows, int cols) nogil:
    cdef int i, j
    cdef double temp
    for i in range(rows):
        temp = 0
        for j in range(cols):
            temp += M[i * cols + j] * v[j]
        r[i] = temp

# Inverse of a matrix - Gaussian Jordan Elimination Method
# Limitations - Expensive for large matrices! O(n3)
cdef int inverse(double* A, double* A_inv, int n) nogil:
    cdef int i, j, k, maxRow
    cdef double maxEl, tmp

    cdef double* aug = <double*>malloc(n * n * 2 * sizeof(double))
    if not aug:
        return -1

    for i in range(n):
        for j in range(n):
            aug[i * 2 * n + j] = A[i * n + j]
            aug[i * 2 * n + j + n] = 1.0 if i == j else 0.0

    for i in range(n):
        maxEl = fabs(aug[i * 2 * n + i])
        maxRow = i
        for k in range(i+1, n):
            if fabs(aug[k * 2 * n + i]) > maxEl:
                maxEl = fabs(aug[k * 2 * n + i])
                maxRow = k
        if maxEl < 1e-12:
            free(aug)
            return -1

        for j in range(2 * n):
            tmp = aug[i * 2 * n + j]
            aug[i * 2 * n + j] = aug[maxRow * 2 * n + j]
            aug[maxRow * 2 * n + j] = tmp

        tmp = aug[i * 2 * n + i]
        for j in range(2 * n):
            aug[i * 2 * n + j] /= tmp

        for k in range(n):
            if k != i:
                tmp = aug[k * 2 * n + i]
                for j in range(2 * n):
                    aug[k * 2 * n + j] -= aug[i * 2 * n + j] * tmp

    for i in range(n):
        for j in range(n):
            A_inv[i * n + j] = aug[i * 2 * n + j + n]

    free(aug)
    return 0

cdef class LinearRegressor:
    cdef int n_features
    cdef double* beta
    cdef double alpha

    def __init__(self, double alpha=1e-8):
        self.n_features = 0
        self.beta = <double*>NULL
        self.alpha = alpha

    def __dealloc__(self):
        if self.beta:
            free(self.beta)

    cpdef list fit(self, list X_py, list y_py):
        cdef int n_samples = len(X_py)

        if n_samples != len(y_py):
            raise ValueError("X and y must have the same number of samples")

        cdef int raw_features = len(X_py[0])
        for row in X_py:
            if len(row) != raw_features:
                raise ValueError("All rows in X must have the same number of features")

        cdef int n_features = raw_features + 1
        self.n_features = n_features

        cdef int i, j
        cdef double* X = <double*>malloc(n_samples * n_features * sizeof(double))
        cdef double* y = <double*>malloc(n_samples * sizeof(double))
        cdef double* X_t = <double*>malloc(n_features * n_samples * sizeof(double))
        cdef double* X_t_X = <double*>malloc(n_features * n_features * sizeof(double))
        cdef double* X_t_y = <double*>malloc(n_features * sizeof(double))
        cdef double* X_t_X_inv = <double*>malloc(n_features * n_features * sizeof(double))

        if not X or not y or not X_t or not X_t_X or not X_t_y or not X_t_X_inv:
            free(X); free(y); free(X_t); free(X_t_X); free(X_t_y); free(X_t_X_inv)
            raise MemoryError("Allocation failed")

        if self.beta:
            free(self.beta)
        self.beta = <double*>malloc(n_features * sizeof(double))
        if not self.beta:
            raise MemoryError("Beta allocation failed")

        try:
            for i in range(n_samples):
                X[i * n_features + 0] = 1.0  # Intercept
                for j in range(raw_features):
                    X[i * n_features + j + 1] = X_py[i][j]
                y[i] = y_py[i]

            transpose_inplace(X, X_t, n_samples, n_features)
            matmul_c(X_t, X, X_t_X, n_features, n_samples, n_features)

            # Ridge regularization to diagonal of X_t_X
            for i in range(n_features):
                X_t_X[i * n_features + i] += self.alpha
            
            matvec_c(X_t, y, X_t_y, n_features, n_samples)

            if inverse(X_t_X, X_t_X_inv, n_features) != 0:
                raise ValueError("Matrix not invertible")

            matvec_c(X_t_X_inv, X_t_y, self.beta, n_features, n_features)

            return [self.beta[i] for i in range(n_features)]

        finally:
            free(X); free(y); free(X_t); free(X_t_X); free(X_t_y); free(X_t_X_inv)

    cpdef list coefficients(self):
        return [self.beta[i] for i in range(self.n_features)]

    cpdef list predict(self, list X_py):
        cdef int n_samples = len(X_py)
        cdef int raw_features = self.n_features - 1
        cdef int i, j

        cdef double* X = <double*>malloc(n_samples * self.n_features * sizeof(double))
        cdef double* y_pred = <double*>malloc(n_samples * sizeof(double))

        if not X or not y_pred:
            free(X); free(y_pred)
            raise MemoryError("Allocation failed")

        try:
            for i in range(n_samples):
                X[i * self.n_features + 0] = 1.0
                for j in range(raw_features):
                    X[i * self.n_features + j + 1] = X_py[i][j]

            matvec_c(X, self.beta, y_pred, n_samples, self.n_features)
            return [y_pred[i] for i in range(n_samples)]

        finally:
            free(X); free(y_pred)

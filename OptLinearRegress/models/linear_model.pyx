# OptLinearRegress/models/linear_model.pyx
# cython: language_level=3
# cython: boundscheck=False, wraparound=False, cdivision=True
# distutils: language = c++

from libc.stdlib cimport malloc, free
from OptLinearRegress.solvers.normal_equation cimport c_solve_normal_equation

cdef class LinearRegressor:
    def __init__(self, double alpha=1e-8):
        self.n_features = 0
        self.beta = <double*>NULL
        self.alpha = alpha    # A small value for convergence

    def __dealloc__(self):
        if self.beta != <double*>NULL:
            free(self.beta)

    cpdef list fit(self, list X_py, list y_py):
        cdef int n_samples = len(X_py)
        cdef int raw_features = len(X_py[0])
        cdef int n_features = raw_features + 1
        self.n_features = n_features

        cdef int i, j, err
        cdef double* X = <double*>malloc(n_samples * n_features * sizeof(double))
        cdef double* y = <double*>malloc(n_samples * sizeof(double))

        if X == NULL or y == NULL:
            if X != NULL: free(X)
            if y != NULL: free(y)
            raise MemoryError("Allocation failed")

        if self.beta != <double*>NULL:
            free(self.beta)
        self.beta = <double*>malloc(n_features * sizeof(double))
        if self.beta == NULL:
            free(X)
            free(y)
            raise MemoryError("Beta allocation failed")

        try:
            for i in range(n_samples):
                X[i * n_features + 0] = 1.0
                for j in range(raw_features):
                    X[i * n_features + j + 1] = X_py[i][j]
                y[i] = y_py[i]

            err = c_solve_normal_equation(X, y, n_samples, n_features, self.alpha, self.beta)
            if err == -1:
                raise ValueError("Matrix not invertible")
            elif err == -2:
                raise MemoryError("Allocation failed")

            return [self.beta[i] for i in range(n_features)]
        finally:
            free(X)
            free(y)

    cpdef list coefficients(self):
        return [self.beta[i] for i in range(self.n_features)]

    cpdef list predict(self, list X_py):
        cdef int n_samples = len(X_py)
        cdef int raw_features = self.n_features - 1
        cdef int i, j
        cdef double temp

        cdef double* X = <double*>malloc(n_samples * self.n_features * sizeof(double))
        cdef double* y_pred = <double*>malloc(n_samples * sizeof(double))

        if X == NULL or y_pred == NULL:
            if X != NULL: free(X)
            if y_pred != NULL: free(y_pred)
            raise MemoryError("Allocation failed")

        try:
            for i in range(n_samples):
                X[i * self.n_features + 0] = 1.0
                for j in range(raw_features):
                    X[i * self.n_features + j + 1] = X_py[i][j]

                temp = 0
                for j in range(self.n_features):
                    temp += X[i * self.n_features + j] * self.beta[j]
                y_pred[i] = temp

            return [y_pred[i] for i in range(n_samples)]
        finally:
            free(X)
            free(y_pred)

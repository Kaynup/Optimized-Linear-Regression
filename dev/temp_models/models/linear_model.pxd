# OptLinearRegress/models/linear_model.pxd
# cython: language_level=3
# cython: boundscheck=False, wraparound=False, cdivision=True

cdef class LinearRegressor:
    cdef int n_features
    cdef double* beta
    cdef double alpha

    cpdef list fit(self, list X_py, list y_py)
    cpdef list coefficients(self)
    cpdef list predict(self, list X_py)

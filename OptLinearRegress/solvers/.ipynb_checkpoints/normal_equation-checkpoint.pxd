# OptLinearRegress/solvers/normal_equation.pxd
# cython: language_level=3

cdef int solve_normal_equation(
    double* X,
    double* y,
    int n_samples,
    int n_features,
    double alpha,
    double* beta
)

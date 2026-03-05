# Solvers API Reference

## Overview

The `solvers` module provides the core numerical optimization algorithms for solving linear regression problems. Specifically, it implements the **Normal Equation** method for least squares solution.

## Normal Equation Solver

### Function: solve_normal_equation

Solves the linear system $(X^T X + \lambda I)\beta = X^T y$ for $\beta$ using matrix inversion.

**Signature**:
```python
def solve_normal_equation(X_flat: memoryview[double],
                          y: memoryview[double],
                          n_samples: int,
                          n_features: int,
                          alpha: float,
                          beta: memoryview[double]) -> int:
```

**Parameters**:
- `X_flat`: Flattened design matrix (memoryview)
  - Shape: `(n_samples * n_features,)`
  - Type: `double[:]` (1D memoryview)
  - Must be allocated and filled before calling
  - Expected layout: row-major (C-contiguous)
  
- `y`: Target value vector (memoryview)
  - Shape: `(n_samples,)`
  - Type: `double[:]`
  
- `n_samples`: Number of training samples
  - Type: `int`
  
- `n_features`: Number of features (including intercept)
  - Type: `int`
  - Typically `original_features + 1` for intercept
  
- `alpha`: L2 regularization parameter (lambda)
  - Type: `float`
  - Typical values: `1e-8` to `1.0`
  
- `beta`: Output coefficient vector (memoryview)
  - Shape: `(n_features,)`
  - Type: `double[:]` (pre-allocated)
  - Will be filled with learned coefficients

**Returns**:
- `0`: Success
- `-1`: Matrix singular (non-invertible)
  - Increase `alpha` if this occurs
- `-2`: Memory allocation failure
  - System ran out of memory

**Time Complexity**: $O(nm^2 + m^3)$
  - Component breakdown:
    - Gram matrix computation: $O(nm^2)$
    - Vector computation: $O(nm)$
    - Matrix inversion: $O(m^3)$
  - Dominated by: $O(m^3)$ for typical high-dimensional problems

**Space Complexity**: $O(m^2)$
  - Gram matrix: $m \times m$
  - Temporary identity matrix during inversion: $m \times m$
  - Note: Input arrays are not copied

**Algorithm Details**:

The solver performs the following steps:

1. **Compute Gram Matrix** ($X^T X + \lambda I$)
   ```
   G[i,j] = sum_{k=1}^{n} X[k,i] * X[k,j]  for i != j
   G[i,i] = sum_{k=1}^{n} X[k,i]^2 + alpha
   ```
   - Time: $O(nm^2)$
   - Space: $O(m^2)$

2. **Compute Cross-Product Vector** ($X^T y$)
   ```
   c[i] = sum_{k=1}^{n} X[k,i] * y[k]
   ```
   - Time: $O(nm)$
   - Space: $O(m)$

3. **Invert Gram Matrix** (Gauss-Jordan elimination)
   - Time: $O(m^3)$
   - Space: $O(m^2)$ for auxiliary identity matrix
   - See [Linear Algebra API](linalg.md) for details

4. **Compute Coefficients** ($\beta = G^{-1} c$)
   ```
   beta[i] = sum_{j=1}^{n_features} G_inv[i,j] * c[j]
   ```
   - Time: $O(m^2)$
   - Space: $O(m)$

**Raises**:
- Returns `-2` on memory allocation failure (wrapped as `MemoryError`)
- Returns `-1` on singular matrix (wrapped as `ValueError`)

### Low-Level C Function: c_solve_normal_equation

**Internal Signature**:
```c
int c_solve_normal_equation(double* X,
                            double* y,
                            int n_samples,
                            int n_features,
                            double alpha,
                            double* beta);
```

This is the internal C-level function called by the Python wrapper. Direct use is not recommended.

**Array Layout**:
- X is row-major: `X[i*n_features + j]` = feature j of sample i
- All computations done in-place where possible

### Usage Example

```python
from OptLinearRegress.solvers import solve_normal_equation
import array

# Prepare data as memoryviews
n_samples = 4
n_features = 3  # 2 actual features + 1 intercept

X_data = array.array('d', [
    1.0, 1.0, 2.0,  # sample 0: intercept=1, x1=1, x2=2
    1.0, 2.0, 3.0,  # sample 1
    1.0, 3.0, 4.0,  # sample 2
    1.0, 4.0, 5.0   # sample 3
])

y_data = array.array('d', [3.0, 5.0, 7.0, 9.0])
beta = array.array('d', [0.0] * n_features)

# Solve
alpha = 1e-8
status = solve_normal_equation(X_data, y_data, n_samples, n_features, alpha, beta)

if status == 0:
    print("Learned coefficients:", list(beta))
elif status == -1:
    print("Singular matrix - increase alpha")
elif status == -2:
    print("Memory allocation failed")
```

### High-Level Usage (Recommended)

With `LinearRegressor` class (recommended approach):

```python
from OptLinearRegress.models import LinearRegressor

model = LinearRegressor(alpha=1e-8)
coeffs = model.fit(X_train, y_train)
```

The model handles all memory management and solver invocation internally.

## Regularization (Alpha Parameter)

The regularization parameter `alpha` affects the solution significantly:

### Gram Matrix Ridge Modification

The matrix becomes: $G = X^T X + \alpha I$ instead of just $X^T X$

**Effect of Alpha**:

| Alpha | Effect | Use Case |
|-------|--------|----------|
| `1e-8` | Near pure least squares | Well-conditioned problems |
| `1e-6` | Minimal regularization | Most cases (recommended start) |
| `1e-4` | Lightweight regularization | Slight numerical instability |
| `0.01` | Moderate regularization | Moderate ill-conditioning |
| `0.1` | Strong regularization | Ill-conditioned or noisy data |
| `1.0` | Very strong regularization | Heavily collinear features |

### Singular Matrix Handling

If the Gram matrix is singular (-1 return):

1. Increase `alpha` incrementally
2. Check for linearly dependent features
3. Remove redundant features
4. Example progression:
   ```python
   alphas = [1e-8, 1e-6, 1e-4, 0.01, 0.1, 1.0]
   for alpha in alphas:
       try:
           model = LinearRegressor(alpha=alpha)
           model.fit(X_train, y_train)
           print(f"Success with alpha={alpha}")
           break
       except ValueError:
           continue
   ```

## Connection to LinearRegressor

The `solve_normal_equation` function is called internally by `LinearRegressor.fit()`:

1. LinearRegressor allocates X (with intercept column)
2. Calls `c_solve_normal_equation` internally
3. Stores learned coefficients in `self.beta`

## Performance Characteristics

**Optimal For**:
- Small to medium feature dimensions ($m < 1000$)
- Need for fast, deterministic solution
- Low-iteration requirement (exact solution)

**Suboptimal For**:
- Very high-dimensional data ($m > 10000$)
- Extremely ill-conditioned matrices
- Streaming/online learning scenarios

**Comparison to Alternatives**:

| Method | Time | Better For |
|--------|------|-----------|
| Normal Equation | $O(nm^2 + m^3)$ | General use |
| Cholesky Decomposition | $O(nm^2 + m^3/3)$ | Slightly faster (symmetric) |
| Gradient Descent | $O(nmk)$ | High-dimensional (k=iterations) |
| SGD | $O(bmk)$ | Streaming data (b=batch) |

## Error Handling

```python
from OptLinearRegress.models import LinearRegressor

try:
    model = LinearRegressor(alpha=1e-10)  # Too small alpha
    model.fit(X_ill_conditioned, y)
except ValueError as e:
    print("Singular matrix - use larger alpha")
    model = LinearRegressor(alpha=1e-6)
    model.fit(X_ill_conditioned, y)
except MemoryError as e:
    print("Out of memory - reduce data size")
```

See [Linear Algebra API](linalg.md) for matrix inversion details and [Models API](models.md) for high-level usage.

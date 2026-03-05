# Linear Regression Algorithm Overview

## Problem Definition

Linear regression solves the following optimization problem:

$$\min_{\beta} \sum_{i=1}^{n} (y_i - \beta_0 - \sum_{j=1}^{m} \beta_j x_{i,j})^2 + \lambda \|\beta\|_2^2$$

Where:
- $n$ = number of samples
- $m$ = number of features
- $x_i$ = feature vector for sample $i$
- $y_i$ = target value for sample $i$
- $\beta$ = coefficient vector (parameters to learn)
- $\lambda$ = L2 regularization strength (alpha parameter)

## Mathematical Formulation

### Matrix Form

The linear regression problem in matrix form:

$$\min_{\beta} \|y - X\beta\|_2^2 + \lambda \|\beta\|_2^2$$

Where $X$ is the design matrix with the intercept column prepended.

### Closed-Form Solution (Normal Equation)

The optimal solution is given by:

$$\beta^* = (X^T X + \lambda I)^{-1} X^T y$$

Where:
- $X^T X$ is the Gram matrix ($m \times m$)
- $\lambda I$ is the regularization term (identity matrix scaled by $\lambda$)
- $X^T y$ is the cross-product vector ($m \times 1$)

## Algorithm: Normal Equation Solver

OptLinearRegress uses the **Normal Equation** method, which provides a closed-form analytical solution.

### Step-by-Step Process

1. **Add Intercept**: Prepend a column of 1s to $X$ (feature matrix)
   ```
   X_augmented = [1, x_{1,1}, x_{1,2}, ..., x_{1,m}]
                 [1, x_{2,1}, x_{2,2}, ..., x_{2,m}]
                 ...
                 [1, x_{n,1}, x_{n,2}, ..., x_{n,m}]
   ```

2. **Compute Gram Matrix**: $G = X^T X + \lambda I$ (Matrix multiplication)
   - Time: $O(nm^2)$ - iterating through $n$ samples and computing $m^2$ elements
   - Space: $O(m^2)$ - storing the $m \times m$ matrix

3. **Compute Cross-Product Vector**: $c = X^T y$
   - Time: $O(nm)$ - iterating through $n$ samples and $m$ features
   - Space: $O(m)$ - vector of size $m$

4. **Invert Matrix**: Compute $G^{-1}$ using Gauss-Jordan elimination
   - Time: $O(m^3)$ - cubic complexity in matrix size
   - Space: $O(m^2)$ - storing auxiliary identity matrix

5. **Compute Coefficients**: $\beta = G^{-1} c$
   - Time: $O(m^2)$ - matrix-vector multiplication
   - Space: $O(m)$ - coefficient vector

### Total Complexity

| Operation | Time | Space |
|-----------|------|-------|
| Gram Matrix | $O(nm^2)$ | $O(m^2)$ |
| Cross-Product | $O(nm)$ | $O(m)$ |
| Matrix Inversion | $O(m^3)$ | $O(m^2)$ |
| Coefficient Computation | $O(m^2)$ | $O(m)$ |
| **Total** | **$O(nm^2 + m^3)$** | **$O(m^2)$** |

**Dominant Term**: 
- For small feature sets ($m << n$): $O(nm^2)$ dominates
- For large feature sets ($m \approx n$): $O(m^3)$ dominates

## Prediction

Once coefficients are learned, predictions are computed via:

$$\hat{y}_i = \beta_0 + \sum_{j=1}^{m} \beta_j x_{i,j}$$

For $k$ prediction samples:
- **Time**: $O(km)$ - dot product for each sample
- **Space**: $O(k)$ - storage for predictions

## Regularization (Alps Parameter)

The regularization term $\lambda I$ provides:

1. **Numerical Stability**: Prevents singular matrices (division issues)
2. **Overfitting Prevention**: Penalizes large coefficient magnitudes
3. **Shrinkage Effect**: Drives coefficients toward zero

### Effect of Different $\lambda$ Values

- **$\lambda = 0$**: Pure least squares (no regularization) - may be numerically unstable
- **$\lambda = 10^{-8}$** (default): Minimal regularization, nearly pure least squares
- **$\lambda = 0.1$**: Moderate regularization, balances fit and simplicity
- **$\lambda = 1.0$**: Strong regularization, heavily shrinks coefficients

## Advantages & Disadvantages

### Advantages [+]
- **Analytical**: Provides exact closed-form solution
- **Fast**: $O(m^3)$ is manageable for typical feature dimensions
- **Stable**: Regularization term improves numerical stability
- **No Iterations**: Unlike gradient descent, no iterative refinement needed
- **Low Memory**: Doesn't require storing all predictions

### Disadvantages [-]
- **Cubic Complexity**: $O(m^3)$ becomes slow for very high-dimensional data
- **Dense Computation**: Matrix inversion is dense, not sparse
- **Memory for Inversion**: Requires $O(m^2)$ temporary space for matrix operations
- **Not Suitable for Streaming**: Requires seeing all data at once

## Comparison with Other Methods

| Method | Time | Space | Stability | Iterations |
|--------|------|-------|-----------|-----------|
| Normal Equation | $O(nm^2 + m^3)$ | $O(m^2)$ | Stable | 0 |
| Gradient Descent | $O(nmk)$ | $O(n+m)$ | Moderate | $k$ |
| SGD | $O(bmk)$ | $O(b+m)$ | Less Stable | $k$ |
| Cholesky | $O(nm^2 + m^3/3)$ | $O(m^2)$ | More Stable | 0 |

Where $k$ = number of iterations, $b$ = batch size

## Implementation Notes

OptLinearRegress implements:
1. **Prime decomposition** of the X matrix computation
2. **Gauss-Jordan elimination** for matrix inversion
3. **In-place operations** to minimize memory allocations
4. **Cython/C++** for performance-critical sections

For more details, see [Normal Equation Solver](normal_equation.md) and [Complexity Analysis](../complexity/time_complexity.md).

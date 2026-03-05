# The Normal Equation: Theory and Practice

## Mathematical Foundations

### Problem Statement

We want to find coefficients $\beta = [\beta_0, \beta_1, ..., \beta_m]$ that minimize squared errors:

$$\min_\beta \sum_{i=1}^{n} (y_i - (\beta_0 + \sum_{j=1}^{m} \beta_j x_{ij}))^2$$

In matrix form:
$$\min_\beta \|Y - X\beta\|_2^2$$

Where:
- $X \in \mathbb{R}^{n \times (m+1)}$ - Feature matrix (first column is 1s for intercept)
- $Y \in \mathbb{R}^n$ - Target vector
- $\beta \in \mathbb{R}^{m+1}$ - Coefficient vector to learn

### Derivation

To find the minimum, we take the derivative with respect to $\beta$ and set to zero:

$$\frac{\partial}{\partial \beta} \|Y - X\beta\|_2^2 = 0$$

$$2X^T(X\beta - Y) = 0$$

$$X^T X \beta = X^T Y$$

This is the **Normal Equation**.

### Solution

If $X^T X$ is invertible:
$$\beta = (X^T X)^{-1} X^T Y$$

This gives the **exact closed-form solution** to the least squares problem.

## Why OptLinearRegress Uses Normal Equation

| Aspect | Normal Equation | Gradient Descent |
|--------|---|---|
| Solution | Exact, closed-form | Approximation, iterative |
| Convergence | 0 iterations | k iterations |
| Time per iteration | $O(nm^2 + m^3)$ | $O(nm)$ |
| Total time | $O(nm^2 + m^3)$ | $O(nmk)$ |
| Memory | $O(m^2)$ | $O(nm)$ or less |
| Feature dimension | <1000 best | >5000 better |
| Suitability | Small to medium m | High-dimensional |

**OptLinearRegress chose Normal Equation for**:
1. No iterations needed (simpler interface)
2. Better for reasonable feature dimensions
3. Exact solution (no convergence uncertainty)
4. Easier to analyze and understand

## Regularization: Ridge Regression

### The Problem: Singular Matrices

When $X^T X$ is singular (non-invertible), we can't compute the inverse:

```
X^T X might be singular when:
- Features are linearly dependent
- More features than samples (m > n)
- Numerical precision limits
```

### The Solution: Add Regularization Term

Instead of solving $X^T X \beta = X^T Y$, we solve:

$$(X^T X + \lambda I) \beta = X^T Y$$

Where:
- $\lambda$ = Regularization strength (our `alpha` parameter)
- $I$ = Identity matrix

This is called **Ridge Regression** or **L2 Regularization**.

### Effect of $\lambda$

**Mathematical effect**:
- Makes $X^T X + \lambda I$ more diagonally dominant
- Diagonal elements increased: $X^T X_{ii} + \lambda$
- Non-singular even if $ X^T X$ was singular!

**Statistical effect**:
- Shrinks coefficients toward zero
- Trade-off between fit (small $\lambda$) and stability (large $\lambda$)
- Reduces overfitting!

### Solution with Regularization

$$\beta = (X^T X + \lambda I)^{-1} X^T Y$$

This is always solvable (as long as $\lambda > 0$).

## Computational Steps

OptLinearRegress solves this in 4 steps:

### Step 1: Compute Gram Matrix
$$G = X^T X + \lambda I$$

Time: $O(nm^2)$

```
for i in range(m):
    for j in range(m):
        G[i,j] = sum(X[k,i] * X[k,j] for k in range(n))
        if i == j:
            G[i,i] += lambda
```

### Step 2: Compute Cross-Product
$$c = X^T Y$$

Time: $O(nm)$

```
for i in range(m):
    c[i] = sum(X[k,i] * Y[k] for k in range(n))
```

### Step 3: Invert Gram Matrix
$$G^{-1} \text{ via Gauss-Jordan elimination}$$

Time: $O(m^3)$

Uses elementary row operations to transform $[G | I] \to [I | G^{-1}]$

### Step 4: Multiply for Solution
$$\beta = G^{-1} c$$

Time: $O(m^2)$

### Total

**Time**: $O(nm^2) + O(nm) + O(m^3) + O(m^2) = O(nm^2 + m^3)$

**Space**: Gram matrix ($m^2$) and identity matrix ($m^2$) = $O(m^2)$

## Choosing Regularization Strength

The `alpha` parameter controls $\lambda$:

### Default: `alpha=1e-8`

```python
model = LinearRegressor()  # Uses alpha=1e-8
```

Nearly pure least squares with minimal numerical stabilization.

**Use when**:
- Data is well-conditioned
- Features are not correlated
- You want minimal bias

**Signs of "too small"**:
```
ERROR: Matrix not invertible
```
→ Increase alpha

### Moderate: `alpha=1e-4` to `alpha=0.1`

```python
model = LinearRegressor(alpha=0.01)
```

Balances fit and regularization.

**Use when**:
- Data has moderate numerical issues
- You want some shrinkage
- Features have some correlation

### Strong: `alpha=1.0` or higher

```python
model = LinearRegressor(alpha=1.0)
```

Heavy regularization, strong shrinkage.

**Use when**:
- Many correlated features
- Highly ill-conditioned matrix
- Preventing overfitting is critical

## Understanding the Bias-Variance Tradeoff

```
Error
  ↑
  │        Total Error
  │       /
  │      /  
  │     /  Bias term ╱
  │    ╱          ╱
  │───────────────  Variance term
  │     ╲______╱
  │          λ (regularization)
  └─────────────────→
```

**Small λ**:
- Low bias (good fit)
- High variance (noisy, unstable)

**Large λ**:
- High bias (underfitting)
- Low variance (smooth, stable)

**Optimal λ**: Balances bias and variance

## Practical Effects Pattern

### Example: House Price Model

```python
from OptLinearRegress.models import LinearRegressor

X_train = # ... 1000 house samples, 50 features ...
y_train = # ... house prices ...
X_test = # ... 200 test samples ...
y_test = # ... test prices ...

alphas = [0, 1e-8, 1e-6, 1e-4, 0.01, 0.1, 1.0]

for alpha in alphas:
    model = LinearRegressor(alpha=alpha)
    model.fit(X_train, y_train)
    
    train_r2 = r2_score(y_train, model.predict(X_train))
    test_r2 = r2_score(y_test, model.predict(X_test))
    
    print(f"α={alpha:.0e}: Train={train_r2:.4f}, Test={test_r2:.4f}")
```

**Expected output**:
```
α=0.0e+00: Train=0.9500, Test=0.7800  (overfitting: 0.17 gap)
α=1.0e-08: Train=0.9500, Test=0.7850  (still overfitting)
α=1.0e-06: Train=0.9450, Test=0.8200  ← better!
α=1.0e-04: Train=0.9300, Test=0.8450  ← even better!
α=1.0e-02: Train=0.8900, Test=0.8620  (good balance)
α=1.0e-01: Train=0.8100, Test=0.8300  (slight underfitting)
α=1.0e+00: Train=0.6500, Test=0.6400  (too much regularization)
```

**Optimal** appears to be around $\alpha = 0.01$ (balanced bias-variance)

## Connection to Ridge Regression

This is exactly **Ridge Regression** from statistics:

$$J(\beta) = \|Y - X\beta\|_2^2 + \lambda \|\beta\|_2^2$$

Where:
- First term: Training error (fit quality)
- Second term: Coefficient penalty (regularization)

The solution minimizes both:
- Want small training error
- Want small coefficients

The tradeoff is controlled by $\lambda$ (our `alpha`).

## Connection to Numerical Stability

From numerical analysis perspective, adding $\lambda I$ improves **conditioning**:

**Condition number**: $\kappa(A) = \|A\| \|A^{-1}\|$

Effects:
- Small $\kappa$: Well-conditioned, stable solutions
- Large $\kappa$: Ill-conditioned, small perturbations cause large errors

**Ridge regression reduces**:
$$\kappa(X^T X + \lambda I) < \kappa(X^T X)$$

Larger $\lambda$ → better conditioning → more stable computation

## Relationship to Other Methods

### Connection to Pseudo-Inverse

The regularized solution approximates:
$$\beta = (X^T X + \lambda I)^{-1} X^T Y \approx X^+ Y$$

Where $X^+$ is the pseudoinverse (computed via SVD).

When $\lambda \to 0$: Approaches pseudoinverse solution

### Connection to Bayesian Regression

From Bayesian perspective:
- Regularization = Prior on coefficients
- $\lambda I$ = Gaussian prior on $\beta$
- Solution = Maximum posterior estimate

### Connection to Principal Component Regression

Can be shown that Ridge regression:
- Shrinks less important principal components more
- Keeps important directions
- Automatically feature selection effect

## Troubleshooting Singular Matrices

### Issue: "Matrix not invertible" Error

**Root cause**:
- Features linearly dependent
- More features than samples
- Numerical precision issues

**Solutions** (in order):

1. **Increase alpha** (quickest fix):
   ```python
   model = LinearRegressor(alpha=1e-4)  # Increase from 1e-8
   ```

2. **Check for duplicate features**:
   ```python
   # Remove redundant columns
   ```

3. **Check for constant features**:
   ```python
   # Features with no variance cause problems
   ```

4. **Check feature scaling**:
   ```python
   # Normalize features to similar scales
   ```

5. **Reduce features** (if too many):
   ```python
   # Use feature selection or PCA
   ```

## Advanced: Choosing Alpha Theoretically

### Cross-Validation (Empirical)

```python
# Estimate generalization error for each alpha
best_alpha = None
best_cv_error = float('inf')

for alpha in [1e-8, 1e-4, 0.01, 1.0]:
    cv_error = cross_validate(X, y, alpha)  # k-fold CV
    if cv_error < best_cv_error:
        best_alpha = alpha
        best_cv_error = cv_error
```

### GCV (Generalized Cross-Validation)

Theoretical approach without explicit cross-validation:

$$\text{GCV}(\lambda) = \frac{n \|Y - \hat{Y}\|_2^2}{(n - \text{trace}(H))^2}$$

Where $H = X(X^T X + \lambda I)^{-1} X^T$ is hat matrix.

Lower GCV → better $\lambda$

(Not implemented in OptLinearRegress currently)

## Key Takeaways

1. **Normal Equation**: Exact closed-form least squares solution
   - $\beta = (X^T X)^{-1} X^T Y$
   - Time: $O(nm^2 + m^3)$

2. **Ridge Regression**: Regularized version handles singularities
   - $\beta = (X^T X + \lambda I)^{-1} X^T Y$
   - Parameter $\lambda$ = our `alpha`

3. **Bias-Variance Tradeoff**: Control via regularization strength
   - Small alpha: Low bias, high variance (overfitting risk)
   - Large alpha: High bias, low variance (underfitting risk)

4. **Practical Alpha Selection**:
   - Start with `alpha=1e-8`
   - If singular: Increase gradually
   - Cross-validate for optimal value

5. **Numerical Stability**: Core reason for regularization
   - Well-conditioned matrix → stable computation
   - Ill-conditioned matrix → numerical errors

See [Time Complexity](../complexity/time_complexity.md) for computational analysis and [Best Practices](best_practices.md) for alpha selection strategies.

## References

- Boyd & Vandenberghe (2004) - Convex Optimization
- Hastie, Tibshirani & Friedman (2009) - Statistical Learning
- Horn & Johnson (2012) - Matrix Analysis
- Golub & Van Loan (2013) - Matrix Computations

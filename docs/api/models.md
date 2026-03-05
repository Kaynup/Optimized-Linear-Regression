# Models API Reference

## LinearRegressor

The main machine learning model class for linear regression with optional L2 regularization (Ridge Regression).

### Class Definition

```python
class LinearRegressor:
    def __init__(self, alpha: float = 1e-8) -> None:
        """
        Initialize a LinearRegressor model.
        
        Parameters
        ----------
        alpha : float, default=1e-8
            L2 regularization strength. Controls the magnitude of the 
            regularization penalty applied to the coefficient vector.
            - alpha=0: Pure least squares (may be numerically unstable)
            - alpha=1e-8: Minimal regularization (default, nearly pure LS)
            - alpha=0.1: Moderate regularization
            - alpha=1.0: Strong regularization
            
        Notes
        -----
        The regularization term is lambda * ||beta||_2^2, added to the
        optimization objective. This implements Ridge Regression.
        
        Examples
        --------
        >>> model = LinearRegressor(alpha=1e-8)
        >>> model = LinearRegressor(alpha=0.1)  # Ridge regression
        """
```

### Methods

#### fit(X_train, y_train)

Fit the linear regression model to training data using the Normal Equation solver.

**Signature**:
```python
def fit(X_train: List[List[float]], 
        y_train: List[float]) -> List[float]:
```

**Parameters**:
- `X_train`: List of lists representing the feature matrix
  - Shape: `(n_samples, n_features)`
  - Type: `list[list[float]]`
  - Each row is a sample, each column is a feature
  
- `y_train`: List of target values
  - Shape: `(n_samples,)`
  - Type: `list[float]`
  - Must have same length as X_train

**Returns**:
- `list[float]`: Learned coefficient vector, length `n_features + 1`
  - First element: intercept (bias term)
  - Remaining elements: coefficients for each feature

**Time Complexity**: $O(nm^2 + m^3)$
  - $n$ = number of training samples
  - $m$ = number of features
  - Dominated by: Matrix inversion ($O(m^3)$) for typical cases

**Space Complexity**: $O(nm + m^2)$
  - Training matrix: $O(nm)$
  - Gram matrix + identity: $O(m^2)$

**Raises**:
- `MemoryError`: If not enough memory to allocate intermediate arrays
- `ValueError`: If the Gram matrix is singular (non-invertible)

**Examples**:
```python
from OptLinearRegress.models import LinearRegressor

# Prepare data
X_train = [[1, 2], [2, 3], [3, 4], [4, 5]]
y_train = [3, 5, 7, 9]

# Fit model
model = LinearRegressor(alpha=1e-8)
coeffs = model.fit(X_train, y_train)
print("Intercept & coefficients:", coeffs)  # [intercept, coeff1, coeff2]
```

**Notes**:
- The intercept is automatically computed and prepended to X internally
- The alpha parameter is crucial for numerical stability
- For singular matrices, increase the alpha value

#### predict(X_test)

Generate predictions on new data using the fitted model.

**Signature**:
```python
def predict(X_test: List[List[float]]) -> List[float]:
```

**Parameters**:
- `X_test`: Feature matrix for which to make predictions
  - Shape: `(n_test, n_features)`
  - Type: `list[list[float]]`
  - Must have same number of features as training data

**Returns**:
- `list[float]`: Predicted target values
  - Shape: `(n_test,)`
  - Predictions from: $\hat{y} = X\beta$

**Time Complexity**: $O(km)$
  - $k$ = number of test samples
  - $m$ = number of features
  - Linear in both dimensions

**Space Complexity**: $O(km)$
  - Test matrix: $O(km)$
  - Prediction vector: $O(k)$

**Raises**:
- `MemoryError`: If not enough memory for prediction arrays

**Examples**:
```python
# Continue from fit example above
X_test = [[2.5, 3.5], [5, 6]]
y_pred = model.predict(X_test)
print("Predictions:", y_pred)  # [4.0, 10.0]
```

**Notes**:
- Model must be fitted before calling predict
- Number of features must match training data

#### coefficients()

Retrieve the learned coefficient vector.

**Signature**:
```python
def coefficients() -> List[float]:
```

**Returns**:
- `list[float]`: Current coefficient vector
  - First element: intercept (bias)
  - Remaining: feature coefficients

**Time Complexity**: $O(m)$ where $m$ = number of features

**Space Complexity**: $O(m)$

**Examples**:
```python
coeffs = model.coefficients()
print(f"Intercept: {coeffs[0]}")
print(f"Feature coefficients: {coeffs[1:]}")
```

**Notes**:
- Returns an empty list if model hasn't been fitted yet
- This is a getter - modifications won't affect predictions

## Parameter Effects

### Effect of Alpha (Regularization)

```python
# Different alpha values for the same data
X = [[1, 2], [2, 3], [3, 4], [4, 5]]
y = [3, 5, 7, 9]

# Minimal regularization (nearly pure least squares)
model1 = LinearRegressor(alpha=1e-8)
model1.fit(X, y)
print(model1.coefficients())  # Large coefficients possible

# Moderate regularization (Ridge regression)
model2 = LinearRegressor(alpha=0.1)
model2.fit(X, y)
print(model2.coefficients())  # Smaller, shrunk coefficients

# Strong regularization
model3 = LinearRegressor(alpha=1.0)
model3.fit(X, y)
print(model3.coefficients())  # Very small coefficients
```

## Common Patterns

### Cross-Validation
```python
from OptLinearRegress.utils.data import train_test_split
from OptLinearRegress.utils.metrics import r2_score

X = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]]
y = [3, 5, 7, 9, 11, 13]

X_train, y_train, X_test, y_test = train_test_split(
    X, y, test_size=0.3, seed=42
)

model = LinearRegressor(alpha=1e-8)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
score = r2_score(y_test, y_pred)
print(f"R² Score: {score}")
```

### Hyperparameter Tuning
```python
# Find best alpha value
best_alpha = 1e-8
best_score = -float('inf')

for alpha in [1e-8, 1e-6, 1e-4, 0.01, 0.1, 1.0]:
    model = LinearRegressor(alpha=alpha)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = r2_score(y_test, y_pred)
    
    if score > best_score:
        best_score = score
        best_alpha = alpha

print(f"Best alpha: {best_alpha}, Score: {best_score}")
```

## Inheritance Notes

LinearRegressor is implemented as a Cython `cdef class`, providing:
- Type safety through Cython compilation
- Memory management via `__dealloc__`
- Direct C++ interoperability
- Automatic garbage collection of internal buffers

See [Solvers API](solvers.md) for the underlying solver implementation.

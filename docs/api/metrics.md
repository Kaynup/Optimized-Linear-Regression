# Metrics API Reference

## Overview

The `metrics` module provides evaluation functions for assessing the performance of linear regression models. All metrics operate on Python lists of predictions vs true values.

## Regression Metrics

### Mean Squared Error (MSE)

Measures the average squared difference between predictions and true values.

**Formula**:
$$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

**Function Signature**:
```python
def mean_squared_error(y_true: List[float], 
                       y_pred: List[float]) -> float:
```

**Parameters**:
- `y_true`: Ground truth target values
  - Type: `list[float]`
  - Length: $n$
  
- `y_pred`: Predicted values
  - Type: `list[float]`
  - Length: $n$ (must match `y_true`)

**Returns**:
- `float`: MSE value (non-negative)
- `0.0`: If no samples or perfect predictions

**Time Complexity**: $O(n)$
- Single pass through both lists

**Space Complexity**: $O(1)$
- Accumulator only

**Interpretation**:
- **MSE = 0**: Perfect predictions
- **Lower MSE**: Better model fit
- **Units**: Squared units of target variable
- **Sensitive to outliers**: Large errors are heavily penalized

**Example**:
```python
from OptLinearRegress.utils.metrics import mean_squared_error

y_true = [1.0, 2.0, 3.0, 4.0]
y_pred = [1.1, 1.9, 3.2, 3.8]

mse = mean_squared_error(y_true, y_pred)
print(f"MSE: {mse:.4f}")  # MSE: 0.0150
```

### Mean Absolute Error (MAE)

Measures the average absolute difference between predictions and true values.

**Formula**:
$$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

**Function Signature**:
```python
def mean_absolute_error(y_true: List[float],
                        y_pred: List[float]) -> float:
```

**Parameters**:
- `y_true`: Ground truth target values
  - Type: `list[float]`
  
- `y_pred`: Predicted values
  - Type: `list[float]`

**Returns**:
- `float`: MAE value (non-negative)
- `0.0`: If no samples

**Time Complexity**: $O(n)$

**Space Complexity**: $O(1)$

**Interpretation**:
- **Units**: Same as target variable
- **Robust to outliers**: Linear penalty for errors
- **Intuitive**: Average magnitude of errors
- **Better for outliers**: Use instead of MSE if data has outliers

**Example**:
```python
from OptLinearRegress.utils.metrics import mean_absolute_error

mae = mean_absolute_error(y_true, y_pred)
print(f"MAE: {mae:.4f}")  # MAE: 0.1000
```

### Root Mean Squared Error (RMSE)

Square root of MSE, in original units.

**Formula**:
$$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$

**Function Signature**:
```python
def root_mean_squared_error(y_true: List[float],
                            y_pred: List[float]) -> float:
```

**Parameters**:
- `y_true`: Ground truth values
- `y_pred`: Predicted values

**Returns**:
- `float`: RMSE value (non-negative)

**Time Complexity**: $O(n)$
- MSE computation plus $O(1)$ square root

**Space Complexity**: $O(1)$

**Interpretation**:
- **Units**: Same as target variable
- **Smoother gradient**: Than MSE for optimization
- **Less sensitive than MSE**: But more than MAE
- **Recommended**: Use for interpretability alongside MSE

**Example**:
```python
from OptLinearRegress.utils.metrics import root_mean_squared_error

rmse = root_mean_squared_error(y_true, y_pred)
print(f"RMSE: {rmse:.4f}")  # RMSE: 0.1225
```

### R² Score (Coefficient of Determination)

Proportion of variance explained by the model (0 to 1, higher is better).

**Formula**:
$$R^2 = 1 - \frac{\text{SS}_\text{res}}{\text{SS}_\text{tot}} = 1 - \frac{\sum_{i=1}^n (y_i - \hat{y}_i)^2}{\sum_{i=1}^n (y_i - \bar{y})^2}$$

Where:
- $\text{SS}_\text{res}$ = residual sum of squares
- $\text{SS}_\text{tot}$ = total sum of squares
- $\bar{y}$ = mean of true values

**Function Signature**:
```python
def r2_score(y_true: List[float],
             y_pred: List[float]) -> float:
```

**Parameters**:
- `y_true`: Ground truth values
- `y_pred`: Predicted values

**Returns**:
- `float`: $R^2$ value
- **Range**: $(-\infty, 1]$ (though typically $[0, 1]$)
- `0.0`: If no samples or $\text{SS}_\text{tot} = 0$

**Time Complexity**: $O(n)$
- Two passes: compute mean, then sums

**Space Complexity**: $O(1)$

**Interpretation**:
- **$R^2 = 1$**: Perfect predictions
- **$R^2 = 0$**: Model explains no variance (as good as mean)
- **$R^2 < 0$**: Model worse than predicting mean
- **$R^2 = 0.9$**: Model explains 90% of variance

**Example**:
```python
from OptLinearRegress.utils.metrics import r2_score

r2 = r2_score(y_true, y_pred)
print(f"R² Score: {r2:.4f}")  # R² Score: 0.9975
```

### Explained Variance Score

Proportion of variance not captured by residuals.

**Formula**:
$$\text{EV} = 1 - \frac{\text{Var}(\text{residuals})}{\text{Var}(y_\text{true})}$$

Where:
- Residuals: $r_i = y_i - \hat{y}_i$
- Var(x) = mean of centered squared values

**Function Signature**:
```python
def explained_variance_score(y_true: List[float],
                             y_pred: List[float]) -> float:
```

**Parameters**:
- `y_true`: Ground truth values
- `y_pred`: Predicted values

**Returns**:
- `float`: Explained variance (typically $[0, 1]$)
- `0.0`: If no samples

**Time Complexity**: $O(n)$
- Multiple passes for variance computations

**Space Complexity**: $O(n)$
- Temporary difference array

**Interpretation**:
- Similar to $R^2$ but uses variance instead of sums
- **$< R^2$**: When residuals are biased
- **$= R^2$**: When residuals have mean 0
- **Use case**: When bias in residuals matters

**Example**:
```python
from OptLinearRegress.utils.metrics import explained_variance_score

ev = explained_variance_score(y_true, y_pred)
print(f"Explained Variance: {ev:.4f}")
```

## Usage Patterns

### Model Evaluation Pipeline

```python
from OptLinearRegress.models import LinearRegressor
from OptLinearRegress.utils.data import train_test_split
from OptLinearRegress.utils.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    root_mean_squared_error
)

# Prepare data
X = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]]
y = [3, 5, 7, 9, 11]

# Split
X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.3)

# Train
model = LinearRegressor(alpha=1e-8)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
metrics = {
    'MSE': mean_squared_error(y_test, y_pred),
    'MAE': mean_absolute_error(y_test, y_pred),
    'RMSE': root_mean_squared_error(y_test, y_pred),
    'R²': r2_score(y_test, y_pred)
}

for name, value in metrics.items():
    print(f"{name}: {value:.4f}")
```

### Cross-Validation

```python
def cross_validate(X, y, alpha=1e-8, n_folds=5):
    from OptLinearRegress.utils.data import train_test_split
    
    fold_size = len(X) // n_folds
    scores = []
    
    for fold in range(n_folds):
        # Create fold split
        test_start = fold * fold_size
        test_end = test_start + fold_size
        
        X_test = X[test_start:test_end]
        y_test = y[test_start:test_end]
        X_train = X[:test_start] + X[test_end:]
        y_train = y[:test_start] + y[test_end:]
        
        # Train model
        model = LinearRegressor(alpha=alpha)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        score = r2_score(y_test, y_pred)
        scores.append(score)
    
    import statistics
    return {
        'mean': statistics.mean(scores),
        'stdev': statistics.stdev(scores) if len(scores) > 1 else 0
    }
```

### Hyperparameter Search

```python
best_alpha = 1e-8
best_score = -float('inf')

alphas = [1e-8, 1e-6, 1e-4, 0.01, 0.1, 1.0]

for alpha in alphas:
    model = LinearRegressor(alpha=alpha)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = r2_score(y_test, y_pred)
    
    print(f"Alpha: {alpha:.0e}, R²: {score:.4f}")
    
    if score > best_score:
        best_score = score
        best_alpha = alpha

print(f"Best alpha: {best_alpha:.0e} (R²: {best_score:.4f})")
```

## Metric Selection Guidelines

| Metric | Best For | Notes |
|--------|----------|-------|
| **MSE** | General purpose | Emphasizes large errors |
| **MAE** | Robust evaluation | Less sensitive to outliers |
| **RMSE** | Interpretability | Same units as target |
| **R²** | Model comparison | Normalized (0-1) scale |
| **EV** | Bias detection | Detects residual bias |

## Complexity Summary

| Metric | Time | Space |
|--------|------|-------|
| MSE | $O(n)$ | $O(1)$ |
| MAE | $O(n)$ | $O(1)$ |
| RMSE | $O(n)$ | $O(1)$ |
| R² | $O(n)$ | $O(1)$ |
| EV Score | $O(n)$ | $O(n)$ |

See [Complexity Analysis](../complexity/time_complexity.md) for detailed breakdown and [User Guide](../guide/user_guide.md) for more examples.

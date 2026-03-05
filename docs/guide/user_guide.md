# User Guide

## Table of Contents

1. [Getting Started](#getting-started)
2. [Basic Workflow](#basic-workflow)
3. [Detailed Examples](#detailed-examples)
4. [Advanced Usage](#advanced-usage)
5. [Troubleshooting](#troubleshooting)

## Getting Started

### What is OptLinearRegress?

OptLinearRegress is a high-performance, lightweight linear regression library built for:
- **Embedded Systems**: Minimal dependencies, pure Python runtime
- **Performance**: Cython/C++17 backend for speed
- **Simplicity**: Easy-to-use API inspired by scikit-learn
- **Research**: Complete transparency and complexity analysis

### Installation

See [Installation & Quick Start](quickstart.md) for detailed setup instructions.

Quick start:
```bash
pip install -r requirements.txt
python setup.py build_ext --inplace
```

### First Model in 30 Seconds

```python
from OptLinearRegress.models import LinearRegressor
from OptLinearRegress.utils.metrics import r2_score

# Create sample data
X_train = [[1, 2], [2, 3], [3, 4], [4, 5]]
y_train = [3, 5, 7, 9]

# Train
model = LinearRegressor()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_train)

# Evaluate
print(f"R²: {r2_score(y_train, y_pred):.4f}")
```

## Basic Workflow

### 1. Prepare Your Data

**Data Format Requirements**:
- Features: `list[list[float]]` (2D list of floats)
- Targets: `list[float]` (1D list of floats)
- Both must have same number of samples

```python
# Example: Predicting house prices
X = [
    [1000, 3, 1],  # sq ft, bedrooms, bathrooms
    [1500, 4, 2],
    [2000, 4, 2],
    [2500, 5, 3],
]
y = [200000, 280000, 350000, 420000]  # prices in $
```

**Data Normalization** (optional but recommended):

```python
def normalize_features(X):
    """Normalize features to zero mean, unit variance."""
    n_features = len(X[0])
    means = [sum(row[j] for row in X) / len(X) for j in range(n_features)]
    stds = [
        (sum((row[j] - means[j])**2 for row in X) / len(X))**0.5 
        for j in range(n_features)
    ]
    return [
        [(row[j] - means[j]) / (stds[j] + 1e-8) for j in range(n_features)]
        for row in X
    ], means, stds

X_normalized, means, stds = normalize_features(X)
```

### 2. Split Data

```python
from OptLinearRegress.utils.data import train_test_split

X_train, y_train, X_test, y_test = train_test_split(
    X, y, 
    test_size=0.2,      # 80-20 split
    shuffle=True,       # randomize order
    seed=42             # for reproducibility
)

print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
```

### 3. Train Model

```python
from OptLinearRegress.models import LinearRegressor

# Create model with regularization parameter
model = LinearRegressor(alpha=1e-8)

# Fit to training data
coefficients = model.fit(X_train, y_train)

print("Learned coefficients:")
print(f"  Intercept: {coefficients[0]:.4f}")
print(f"  Coefficients: {coefficients[1:]}")
```

**Understanding Alpha**:
- `alpha=0`: Pure least squares (may be unstable)
- `alpha=1e-8`: Default, minimal regularization
- `alpha=0.1`: Moderate regularization (Ridge regression)
- `alpha=1.0`: Strong regularization

### 4. Make Predictions

```python
# Predict on test set
y_pred = model.predict(X_test)

# Predict on single sample (must still be list of lists)
single_pred = model.predict([[1200, 3, 2]])
print(f"Predicted price: ${single_pred[0]:,.0f}")
```

### 5. Evaluate Performance

```python
from OptLinearRegress.utils.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    root_mean_squared_error
)

# Get predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Compute metrics
print("Training Metrics:")
print(f"  MSE:  {mean_squared_error(y_train, y_train_pred):,.0f}")
print(f"  RMSE: {root_mean_squared_error(y_train, y_train_pred):,.0f}")
print(f"  MAE:  {mean_absolute_error(y_train, y_train_pred):,.0f}")
print(f"  R²:   {r2_score(y_train, y_train_pred):.4f}")

print("\\nTest Metrics:")
print(f"  MSE:  {mean_squared_error(y_test, y_test_pred):,.0f}")
print(f"  RMSE: {root_mean_squared_error(y_test, y_test_pred):,.0f}")
print(f"  MAE:  {mean_absolute_error(y_test, y_test_pred):,.0f}")
print(f"  R²:   {r2_score(y_test, y_test_pred):.4f}")
```

## Detailed Examples

### Example 1: Simple Linear Relationship

```python
from OptLinearRegress.models import LinearRegressor
from OptLinearRegress.utils.metrics import r2_score
import math

# Create perfect linear data: y = 2x + 1
X = [[x] for x in range(1, 11)]
y = [2*x[0] + 1 for x in X]

# Train
model = LinearRegressor(alpha=1e-8)
model.fit(X, y)

# Check learned coefficients
coeffs = model.coefficients()
print(f"Intercept: {coeffs[0]:.4f} (expected: 1)")
print(f"Slope: {coeffs[1]:.4f} (expected: 2)")

# Perfect R²
y_pred = model.predict(X)
print(f"R²: {r2_score(y, y_pred):.4f} (expected: 1.0)")
```

### Example 2: House Price Prediction

```python
from OptLinearRegress.models import LinearRegressor
from OptLinearRegress.utils.data import train_test_split
from OptLinearRegress.utils.metrics import r2_score, mean_absolute_error

# Dataset: square footage, bedrooms, bathrooms -> price
data = [
    ([900, 2, 1], 180000),
    ([1100, 3, 1], 220000),
    ([1300, 3, 2], 260000),
    ([1500, 3, 2], 300000),
    ([1700, 4, 2], 340000),
    ([1900, 4, 3], 380000),
    ([2100, 4, 3], 420000),
    ([2300, 5, 3], 460000),
]

X = [d[0] for d in data]
y = [d[1] for d in data]

# Split & train
X_train, y_train, X_test, y_test = train_test_split(
    X, y, test_size=0.25, seed=42
)

model = LinearRegressor(alpha=1e-5)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MAE: ${mae:,.0f}")
print(f"R²: {r2:.4f}")

# Predict new house
new_house = [[2000, 4, 2.5]]
predicted_price = model.predict(new_house)[0]
print(f"\\nNew house (2000 sqft, 4 bed, 2.5 bath): ${predicted_price:,.0f}")
```

### Example 3: Hyperparameter Tuning (Alpha)

```python
from OptLinearRegress.models import LinearRegressor
from OptLinearRegress.utils.data import train_test_split
from OptLinearRegress.utils.metrics import r2_score

# ... load data ...

X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.2)

# Test different alpha values
alphas = [0, 1e-8, 1e-6, 1e-4, 0.01, 0.1, 1.0]
results = []

for alpha in alphas:
    model = LinearRegressor(alpha=alpha)
    model.fit(X_train, y_train)
    
    train_score = r2_score(y_train, model.predict(X_train))
    test_score = r2_score(y_test, model.predict(X_test))
    
    results.append((alpha, train_score, test_score))

# Find best alpha (highest test score)
best = max(results, key=lambda x: x[2])
print(f"Best alpha: {best[0]} (Test R²: {best[2]:.4f})")

# Print comparison
print("\\nAlpha\\tTrain R²\\tTest R²")
for alpha, train, test in results:
    print(f"{alpha:.0e}\\t{train:.4f}\\t{test:.4f}")
```

### Example 4: Cross-Validation

```python
from OptLinearRegress.models import LinearRegressor
from OptLinearRegress.utils.metrics import r2_score
import statistics

def k_fold_cv(X, y, k=5, alpha=1e-8):
    """Perform k-fold cross-validation."""
    n = len(X)
    fold_size = n // k
    scores = []
    
    for fold in range(k):
        # Split into fold
        test_start = fold * fold_size
        test_end = test_start + fold_size
        
        X_test_fold = X[test_start:test_end]
        y_test_fold = y[test_start:test_end]
        
        X_train_fold = X[:test_start] + X[test_end:]
        y_train_fold = y[:test_start] + y[test_end:]
        
        # Train and evaluate
        model = LinearRegressor(alpha=alpha)
        model.fit(X_train_fold, y_train_fold)
        
        y_pred = model.predict(X_test_fold)
        score = r2_score(y_test_fold, y_pred)
        scores.append(score)
        
        print(f"Fold {fold+1}: R² = {score:.4f}")
    
    return {
        'mean': statistics.mean(scores),
        'stdev': statistics.stdev(scores) if k > 1 else 0,
        'scores': scores
    }

# Run 5-fold CV
cv_results = k_fold_cv(X, y, k=5, alpha=1e-8)
print(f"\\nMean R²: {cv_results['mean']:.4f} ± {cv_results['stdev']:.4f}")
```

## Advanced Usage

### Batch Predictions

```python
from OptLinearRegress.utils.data import batch_iterator

# Process predictions in batches (memory efficient)
def predict_in_batches(model, X, batch_size=1000):
    all_predictions = []
    for X_batch, _ in batch_iterator(X, [0]*len(X), batch_size=batch_size):
        y_batch = model.predict(X_batch)
        all_predictions.extend(y_batch)
    return all_predictions

# Or more simply:
y_test_pred = model.predict(X_test)
```

### Multiple Model Comparison

```python
def compare_alphas(X_train, y_train, X_test, y_test):
    """Compare multiple regularization strengths."""
    from OptLinearRegress.models import LinearRegressor
    from OptLinearRegress.utils.metrics import r2_score, mean_squared_error
    
    alphas = [0, 1e-10, 1e-8, 1e-6, 1e-4, 0.01, 0.1, 1.0]
    
    results = []
    for alpha in alphas:
        try:
            model = LinearRegressor(alpha=alpha)
            model.fit(X_train, y_train)
            
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            results.append({
                'alpha': alpha,
                'train_r2': r2_score(y_train, y_train_pred),
                'test_r2': r2_score(y_test, y_test_pred),
                'train_mse': mean_squared_error(y_train, y_train_pred),
                'test_mse': mean_squared_error(y_test, y_test_pred),
                'overfit': r2_score(y_train, y_train_pred) - r2_score(y_test, y_test_pred)
            })
        except ValueError as e:
            print(f"Alpha {alpha} failed: {e}")
    
    return results

results = compare_alphas(X_train, y_train, X_test, y_test)
for r in results:
    print(f"Alpha={r['alpha']:.0e}: Train R²={r['train_r2']:.4f}, " + 
          f"Test R²={r['test_r2']:.4f}, Overfit={r['overfit']:.4f}")
```

### Feature Importance (via Coefficients)

```python
model = LinearRegressor(alpha=1e-8)
model.fit(X_train, y_train)

coeffs = model.coefficients()
feature_names = ['intercept', 'sqft', 'bedrooms', 'bathrooms']

print("Feature Importance (by coefficient magnitude):")
for name, coeff in zip(feature_names, coeffs):
    if name != 'intercept':
        print(f"  {name:15} {coeff:12.2f}")
```

## Troubleshooting

### Issue: "Matrix not invertible" Error

**Cause**: Singular or near-singular Gram matrix

**Solution**:
```python
# Increase regularization
model = LinearRegressor(alpha=1e-4)  # Increase from 1e-8
model.fit(X_train, y_train)

# Or check for linearly dependent features
# Remove redundant features if possible
```

### Issue: Very Poor Model Performance

**Causes & Solutions**:
1. **Features not normalized**:
   ```python
   X_normalized = normalize_features(X)
   ```

2. **Bad data quality**:
   ```python
   # Remove outliers
   X_clean = [x for x, y in zip(X, y) if abs(y - y_mean) < 3*y_std]
   ```

3. **Wrong features**:
   ```python
   # Do feature analysis
   print("Feature correlations with target...")
   ```

4. **Too much regularization**:
   ```python
   model = LinearRegressor(alpha=0)  # Try minimal alpha
   ```

### Issue: Memory Error on Large Data

**Solutions**:
1. **Use data batching**:
   ```python
   # Process in smaller chunks
   # Note: Can't directly batch fit() with current API
   # But can batch predictions
   y_pred_batches = []
   for X_batch, _ in batch_iterator(X_test, [0]*len(X_test), 1000):
       y_pred_batches.extend(model.predict(X_batch))
   ```

2. **Reduce features**:
   ```python
   # Use fewer, more important features
   X_reduced = [[row[i] for i in important_indices] for row in X]
   ```

3. **Subsample training data**:
   ```python
   # Use random subset
   import random
   indices = random.sample(range(len(X)), k=100000)
   X_subset = [X[i] for i in indices]
   y_subset = [y[i] for i in indices]
   ```

### Issue: Non-Reproducible Results

**Solution**: Set random seed
```python
from OptLinearRegress.utils.data import train_test_split

X_train, y_train, X_test, y_test = train_test_split(
    X, y, seed=42  # Set seed for reproducibility
)
```

## Performance Tips

1. **Minimize features**: Biggest impact on speed (cubic complexity)
2. **Normalize features**: Improves stability and regularization effectiveness
3. **Use appropriate alpha**: Too small causes instability, too large hurts fit
4. **Batch predictions**: For very large test sets, process in batches
5. **Pre-allocate if possible**: Reduces memory allocation overhead

See [Complexity Analysis](../complexity/time_complexity.md) for detailed performance characteristics.

See [Best Practices](best_practices.md) for more optimization strategies.

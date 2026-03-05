# Best Practices

## General Principles

### 1. Always Normalize Features

**Why**: Features are in different scales, which:
- Makes regularization less effective
- Causes numerical instability
- Makes coefficients hard to interpret

```python
def normalize_features(X):
    """Normalize features to zero mean, unit variance."""
    n = len(X)
    m = len(X[0])
    
    # Compute means
    means = [sum(x[j] for x in X) / n for j in range(m)]
    
    # Compute standard deviations
    stds = [(sum((x[j] - means[j])**2 for x in X) / n)**0.5 for j in range(m)]
    
    # Normalize
    return [
        [(x[j] - means[j]) / (stds[j] + 1e-8) for j in range(m)]
        for x in X
    ], means, stds

X_norm, means, stds = normalize_features(X)
model.fit(X_norm, y)

# When predicting on new data, normalize similarly
def apply_normalization(X_new, means, stds):
    return [[(x[j] - means[j]) / stds[j] for j in range(len(x))] for x in X_new]

X_test_norm = apply_normalization(X_test, means, stds)
y_pred = model.predict(X_test_norm)
```

### 2. Use Train-Test Split Consistently

**Why**: Prevents overfitting estimates and gives honest performance assessment.

```python
from OptLinearRegress.utils.data import train_test_split

# Always split before training
X_train, y_train, X_test, y_test = train_test_split(
    X, y,
    test_size=0.2,      # Standard 80-20
    shuffle=True,       # Randomize
    seed=42             # For reproducibility
)

# Train only on training set
model = LinearRegressor(alpha=1e-8)
model.fit(X_train, y_train)

# Evaluate only on test set
y_test_pred = model.predict(X_test)
test_r2 = r2_score(y_test, y_test_pred)
```

### 3. Set Random Seed for Reproducibility

**Why**: Allows others to reproduce your exact results.

```python
from OptLinearRegress.utils.data import train_test_split

# Use seed parameter everywhere random is involved
X_train, y_train, X_test, y_test = train_test_split(
    X, y,
    seed=42  # Makes split reproducible
)
```

### 4. Check for Data Quality Issues

```python
def validate_data(X, y):
    """Check for common data quality problems."""
    issues = []
    
    # Check dimensions
    if len(X) != len(y):
        issues.append("Sample mismatch: len(X) != len(y)")
    
    # Check for NaN/None
    for i, x in enumerate(X):
        if any(v is None or str(v) == 'nan' for v in x):
            issues.append(f"Row {i} has missing values")
            break
    
    # Check for constant features (zero variance)
    m = len(X[0])
    for j in range(m):
        col = [x[j] for x in X]
        if max(col) == min(col):
            issues.append(f"Feature {j} is constant")
    
    # Check for duplicate rows
    unique_rows = len(set(tuple(x) for x in X))
    if unique_rows < len(X):
        issues.append(f"Found {len(X) - unique_rows} duplicate samples")
    
    return issues if issues else ["Data OK"]

for issue in validate_data(X, y):
    print(issue)
```

---

## Regularization (Alpha) Selection

### Strategy 1: Cross-Validation (Recommended)

```python
from OptLinearRegress.models import LinearRegressor
from OptLinearRegress.utils.metrics import r2_score

def find_best_alpha(X_train, y_train, X_val, y_val):
    """Find alpha that minimizes validation error."""
    alphas = [0, 1e-10, 1e-8, 1e-6, 1e-4, 0.01, 0.1, 1.0]
    best_alpha = None
    best_score = -float('inf')
    
    for alpha in alphas:
        try:
            model = LinearRegressor(alpha=alpha)
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_val)
            score = r2_score(y_val, y_pred)
            
            if score > best_score:
                best_score = score
                best_alpha = alpha
                
            print(f"α={alpha:.0e}: validation R² = {score:.4f}")
        except ValueError:
            print(f"α={alpha:.0e}: Singular matrix!")
    
    return best_alpha, best_score

# Split data
X_train, _, X_val, y_val = train_test_split(X, y, test_size=0.3)
best_alpha, _ = find_best_alpha(X_train, y_train, X_val, y_val)

# Retrain on full training set with best alpha
model = LinearRegressor(alpha=best_alpha)
model.fit(X_train, y_train)
```

### Strategy 2: Start Small and Increase

```python
# If you get "Matrix not invertible" error
model = LinearRegressor(alpha=1e-8)   # Default
try:
    model.fit(X_train, y_train)
except ValueError:
    print("Try larger alpha...")
    
    for alpha in [1e-6, 1e-4, 0.01, 0.1, 1.0]:
        try:
            model = LinearRegressor(alpha=alpha)
            model.fit(X_train, y_train)
            print(f"Success with α={alpha}")
            break
        except ValueError:
            continue
```

### Strategy 3: Empirical Testing

```python
# Test different alphas and compare train/test error
alphas_to_test = [1e-8, 1e-4, 0.01, 0.1, 1.0]

for alpha in alphas_to_test:
    model = LinearRegressor(alpha=alpha)
    model.fit(X_train, y_train)
    
    train_r2 = r2_score(y_train, model.predict(X_train))
    test_r2 = r2_score(y_test, model.predict(X_test))
    
    print(f"α={alpha:6.0e} | Train: {train_r2:.4f} | Test: {test_r2:.4f} | Gap: {train_r2-test_r2:.4f}")
    
    # Look for good gap (not overfitting)
```

---

## Feature Selection & Engineering

### Remove Irrelevant Features

```python
def compute_feature_importance(model, X_train, y_train):
    """
    Estimate feature importance via coefficient magnitude.
    Warning: Only valid for normalized features!
    """
    coeffs = model.coefficients()
    # Skip intercept
    feature_coeffs = coeffs[1:]
    
    # Absolute value shows magnitude of effect
    importance = [(i, abs(c)) for i, c in enumerate(feature_coeffs)]
    importance.sort(key=lambda x: x[1], reverse=True)
    
    return importance

model = LinearRegressor(alpha=1e-8)
model.fit(X_train, y_train)

importance = compute_feature_importance(model, X_train, y_train)
for feature_idx, coeff_mag in importance:
    print(f"Feature {feature_idx}: magnitude {coeff_mag:.6f}")

# Keep only top features
top_indices = [i for i, _ in importance[:10]]  # Top 10
X_reduced = [[x[i] for i in top_indices] for x in X]
```

### Feature Engineering: Polynomial Features

```python
def create_polynomial_features(X, degree=2):
    """Create polynomial features up to given degree."""
    X_poly = []
    for x in X:
        features = list(x)  # Original features
        
        if degree >= 2:
            # Quadratic terms: x²
            for xi in x:
                features.append(xi * xi)
            
            # Interaction terms: x_i * x_j
            m = len(x)
            for i in range(m):
                for j in range(i+1, m):
                    features.append(x[i] * x[j])
        
        X_poly.append(features)
    
    return X_poly

# Use polynomial features
X_poly = create_polynomial_features(X, degree=2)
model = LinearRegressor(alpha=1e-8)
model.fit(X_poly, y)
```

---

## Memory Optimization

### For Very Large Datasets

```python
import random

def subsample_data(X, y, max_samples=100000):
    """Reduce dataset while maintaining distribution."""
    if len(X) <= max_samples:
        return X, y
    
    # Random subsample
    indices = random.sample(range(len(X)), max_samples)
    X_sub = [X[i] for i in indices]
    y_sub = [y[i] for i in indices]
    
    return X_sub, y_sub

# For millions of samples
X_sub, y_sub = subsample_data(X, y, max_samples=100000)
model.fit(X_sub, y_sub)

# Prediction still works on full data
y_pred = model.predict(X)
```

### Predict in Batches

```python
def predict_batches(model, X_test, batch_size=10000):
    """Predict on test data in memory-efficient batches."""
    all_predictions = []
    
    for i in range(0, len(X_test), batch_size):
        X_batch = X_test[i:i+batch_size]
        y_batch = model.predict(X_batch)
        all_predictions.extend(y_batch)
    
    return all_predictions

# Use for large test sets
y_pred = predict_batches(model, X_test, batch_size=50000)
```

---

## Error Handling

### Robust Training Pipeline

```python
def train_with_fallback(X_train, y_train, X_test, y_test):
    """Train with fallback in case of numerical issues."""
    
    # Try default alpha
    for alpha in [1e-8, 1e-6, 1e-4, 0.01, 0.1, 1.0]:
        try:
            model = LinearRegressor(alpha=alpha)
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            test_r2 = r2_score(y_test, y_pred)
            
            print(f"[+] Success with α={alpha}: R²={test_r2:.4f}")
            return model, alpha
            
        except ValueError as e:
            print(f"[-] Failed with α={alpha}: {e}")
            continue
    
    raise RuntimeError("Failed to train model with any alpha value")

try:
    model, alpha = train_with_fallback(X_train, y_train, X_test, y_test)
except RuntimeError as e:
    print(f"Error: {e}")
    print("Suggestions:")
    print("  1. Check for duplicate/constant features")
    print("  2. Verify data quality (no NaNs, valid ranges)")
    print("  3. Try feature normalization")
```

---

## Performance Optimization

### Profile Your Code

```python
import time

def time_operation(func, *args, **kwargs):
    """Measure execution time."""
    start = time.time()
    result = func(*args, **kwargs)
    elapsed = time.time() - start
    return result, elapsed

# Time fitting
model = LinearRegressor(alpha=1e-8)
_, fit_time = time_operation(model.fit, X_train, y_train)
print(f"Fitting took {fit_time:.4f} seconds")

# Time prediction
_, pred_time = time_operation(model.predict, X_test)
print(f"Prediction took {pred_time:.4f} seconds")

# Time evaluation
y_pred = model.predict(X_test)
_, eval_time = time_operation(r2_score, y_test, y_pred)
print(f"Evaluation took {eval_time:.6f} seconds")
```

### Know Your Complexity

```python
def estimate_time(n_samples, n_features):
    """
    Rough estimate of fitting time based on complexity.
    Assumes ~1 billion operations/second on modern CPU.
    """
    ops = n_samples * (n_features**2) + n_features**3
    estimated_seconds = ops / 1e9
    
    return estimated_seconds

# Examples
print(f"1k samples, 100 features: {estimate_time(1000, 100):.3f} seconds")
print(f"1m samples, 100 features: {estimate_time(1e6, 100):.3f} seconds")
print(f"1m samples, 1000 features: {estimate_time(1e6, 1000):.1f} seconds")
```

---

## Validation & Testing

### Cross-Validation

```python
def kfold_validation(X, y, k=5, alpha=1e-8):
    """Perform k-fold cross-validation."""
    n = len(X)
    fold_size = n // k
    scores = []
    
    for fold in range(k):
        # Create fold split
        test_start = fold * fold_size
        test_end = test_start + fold_size if fold < k-1 else n
        
        X_val = X[test_start:test_end]
        y_val = y[test_start:test_end]
        X_tr = X[:test_start] + X[test_end:]
        y_tr = y[:test_start] + y[test_end:]
        
        # Train and evaluate
        model = LinearRegressor(alpha=alpha)
        model.fit(X_tr, y_tr)
        
        y_pred = model.predict(X_val)
        score = r2_score(y_val, y_pred)
        scores.append(score)
        
        print(f"Fold {fold+1}: R² = {score:.4f}")
    
    import statistics
    mean_score = statistics.mean(scores)
    std_score = statistics.stdev(scores) if k > 1 else 0
    
    print(f"\nMean R²: {mean_score:.4f} ± {std_score:.4f}")
    return mean_score, std_score

mean, std = kfold_validation(X, y, k=5)
```

### Residual Analysis

```python
def analyze_residuals(y_true, y_pred):
    """Analyze model residuals for issues."""
    residuals = [y_true[i] - y_pred[i] for i in range(len(y_true))]
    
    import statistics
    mean_res = statistics.mean(residuals)
    std_res = statistics.stdev(residuals)
    
    print(f"Residual mean: {mean_res:.6f} (should be ~0)")
    print(f"Residual std: {std_res:.6f}")
    
    # Check for outlier residuals
    outliers = [r for r in residuals if abs(r) > 3*std_res]
    print(f"Outlier residuals (>3σ): {len(outliers)} / {len(residuals)}")
    
    if outliers:
        print("  Warning: Model has outlier predictions")

y_pred = model.predict(X_test)
analyze_residuals(y_test, y_pred)
```

---

## Documentation & Reproducibility

### Document Your Workflow

```python
# Good practice: document everything
"""
House Price Prediction Model

Data:
  - 1000 house samples
  - Features: sqft, bedrooms, bathrooms, age
  - Target: price (USD)

Preprocessing:
  - Normalized all features (zero mean, unit variance)
  - 80-20 train-test split with seed 42
  
Model:
  - Linear Regression with alpha=1e-6
  - Selected via 5-fold cross-validation
  
Results:
  - Train R²: 0.9123
  - Test R²: 0.8891
  - MSE: 2500000
"""

from OptLinearRegress.models import LinearRegressor
from OptLinearRegress.utils.data import train_test_split

# ... training code ...
```

### Save Model Configuration

```python
# Save important metadata
model_metadata = {
    'alpha': 1e-8,
    'n_features': 100,
    'coefficients': model.coefficients(),
    'training_samples': len(X_train),
    'test_r2': 0.9123,
    'feature_names': ['sqft', 'beds', 'baths', ...],
    'feature_means': [2000, 3.5, 2.1, ...],
    'feature_stds': [500, 0.8, 0.6, ...]
}

# Save to JSON for later use
import json
with open('model_config.json', 'w') as f:
    json.dump(model_metadata, f, indent=2)
```

---

## Common Pitfalls to Avoid

| Pitfall | Problem | Solution |
|---------|---------|----------|
| Training on test data | Overfitting estimates | Always split before training |
| Unnormalized features | Instability, poor regularization | Normalize all features |
| No random seed | Non-reproducible results | Set seed in all random operations |
| Too much regularization | Model can't fit | Use cross-validation to tune alpha |
| Too little regularization | Numerical instability | Increase alpha if singular |
| Ignoring missing data | Wrong results | Handle before training |
| Using same data for training and evaluation | False R² | Use separate test set |
| Skipping feature validation | Garbage in, garbage out | Profile features before training |

---

See [Complexity Analysis](../complexity/summary.md) for performance optimization and [Glossary](../glossary.md) for term definitions.

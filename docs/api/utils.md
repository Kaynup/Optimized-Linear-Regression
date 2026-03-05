# Utils API Reference

## Overview

The `utils` module provides utility functions for data handling, preprocessing, and model evaluation.

## Data Utilities (data.py)

### train_test_split

Splits data into training and testing sets with optional shuffling.

**Function Signature**:
```python
def train_test_split(X: List[List[float]],
                     y: List[float],
                     test_size: float = 0.2,
                     shuffle: bool = True,
                     seed: Optional[int] = None
                     ) -> Tuple[List, List, List, List]:
```

**Parameters**:
- `X`: Feature matrix
  - Type: `list[list[float]]`
  - Shape: `(n_samples, n_features)`
  
- `y`: Target vector
  - Type: `list[float]`
  - Shape: `(n_samples,)`
  - Must match length of X
  
- `test_size`: Fraction for test set
  - Type: `float`
  - Range: `(0, 1)` (e.g., 0.2 = 20%)
  - Default: `0.2` (80-20 split)
  
- `shuffle`: Whether to shuffle before splitting
  - Type: `bool`
  - Default: `True`
  
- `seed`: Random seed for reproducibility
  - Type: `int` or `None`
  - Default: `None` (non-deterministic)

**Returns**:
- Tuple of 4 lists:
  ```python
  X_train, y_train, X_test, y_test
  ```
  - `X_train`: Training features (80% of data by default)
  - `y_train`: Training targets
  - `X_test`: Test features (20% of data by default)
  - `y_test`: Test targets

**Time Complexity**: $O(n + s)$ where $s$ = samples shuffled
- Shuffling: $O(n)$
- Copying: $O(n)$

**Space Complexity**: $O(n)$
- Indices and output arrays

**Example**:
```python
from OptLinearRegress.utils.data import train_test_split

X = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]]
y = [1, 2, 3, 4, 5]

# Default 80-20 split, shuffled
X_train, y_train, X_test, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=True, seed=42
)

print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
# Train size: 4, Test size: 1
```

### batch_iterator

Creates batches for mini-batch processing.

**Function Signature**:
```python
def batch_iterator(X: List[List[float]],
                   y: List[float],
                   batch_size: int = 32,
                   shuffle: bool = True,
                   seed: Optional[int] = None
                   ) -> Iterator[Tuple[List, List]]:
```

**Parameters**:
- `X`: Feature matrix
  - Type: `list[list[float]]`
  
- `y`: Target vector
  - Type: `list[float]`
  
- `batch_size`: Samples per batch
  - Type: `int`
  - Default: `32`
  
- `shuffle`: Shuffle data before batching
  - Type: `bool`
  - Default: `True`
  
- `seed`: Random seed
  - Type: `int` or `None`

**Yields**:
- `Tuple[List, List]`: `(X_batch, y_batch)`
  - Each batch contains up to `batch_size` samples
  - Last batch may be smaller

**Time Complexity**: $O(n + n\log n)$
- Shuffling: $O(n\log n)$ worst case
- Yielding: $O(n)$ total across all batches

**Space Complexity**: $O(b + m)$
- `b` = batch_size
- `m` = n_features

**Example**:
```python
from OptLinearRegress.utils.data import batch_iterator

X = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]]
y = [1, 2, 3, 4, 5]

for X_batch, y_batch in batch_iterator(X, y, batch_size=2, seed=42):
    print(f"Batch: X shape {len(X_batch)}, y shape {len(y_batch)}")
    # Batch: X shape 2, y shape 2
    # Batch: X shape 2, y shape 2
    # Batch: X shape 1, y shape 1
```

### shuffle_arrays

Shuffles X and y in parallel with same permutation.

**Function Signature**:
```python
def shuffle_arrays(X: List[List[float]],
                   y: List[float],
                   seed: Optional[int] = None
                   ) -> Tuple[List, List]:
```

**Parameters**:
- `X`: Feature matrix
- `y`: Target vector
- `seed`: Random seed (default: `None`)

**Returns**:
- `Tuple[List, List]`: `(X_shuffled, y_shuffled)`
  - Same permutation applied to both

**Time Complexity**: $O(n\log n)$
- Random shuffle

**Space Complexity**: $O(n)$

**Example**:
```python
from OptLinearRegress.utils.data import shuffle_arrays

X_shuffled, y_shuffled = shuffle_arrays(X, y, seed=42)
```

### add_intercept

Prepends a bias column (column of 1s) to feature matrix.

**Function Signature**:
```python
def add_intercept(X: List[List[float]]) -> List[List[float]]:
```

**Parameters**:
- `X`: Feature matrix
  - Shape: `(n_samples, n_features)`

**Returns**:
- `List[List[float]]`: Feature matrix with intercept
  - Shape: `(n_samples, n_features + 1)`
  - First column: all 1s

**Time Complexity**: $O(nm)$
- Copying and augmentation

**Space Complexity**: $O(nm)$

**Example**:
```python
from OptLinearRegress.utils.data import add_intercept

X = [[1, 2], [3, 4]]
X_with_intercept = add_intercept(X)
# [[1, 1, 2], [1, 3, 4]]
```

**Note**: LinearRegressor automatically adds intercept, so manual use is rarely needed.

## Metrics Utilities

See [Metrics API](metrics.md) for detailed documentation.

Quick reference:
- `mean_squared_error(y_true, y_pred)` → MSE
- `mean_absolute_error(y_true, y_pred)` → MAE
- `root_mean_squared_error(y_true, y_pred)` → RMSE
- `r2_score(y_true, y_pred)` → R²
- `explained_variance_score(y_true, y_pred)` → Explained Variance

## Complete Example: Training Pipeline

```python
from OptLinearRegress.models import LinearRegressor
from OptLinearRegress.utils.data import train_test_split
from OptLinearRegress.utils.metrics import r2_score, mean_squared_error

# 1. Load/prepare data
X = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8]]
y = [3, 5, 7, 9, 11, 13, 15]

# 2. Split data
X_train, y_train, X_test, y_test = train_test_split(
    X, y, test_size=0.3, seed=42
)

# 3. Train model
model = LinearRegressor(alpha=1e-8)
model.fit(X_train, y_train)
print("Coefficients:", model.coefficients())

# 4. Evaluate
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

print(f"Train R²: {r2_score(y_train, y_train_pred):.4f}")
print(f"Test R²: {r2_score(y_test, y_test_pred):.4f}")
print(f"Test MSE: {mean_squared_error(y_test, y_test_pred):.4f}")
```

## Batch Processing Example

```python
from OptLinearRegress.models import LinearRegressor
from OptLinearRegress.utils.data import batch_iterator

# For analyzing predictions in batches
model = LinearRegressor(alpha=1e-8)
model.fit(X_train, y_train)

batch_predictions = []
for X_batch, y_batch in batch_iterator(X_test, y_test, batch_size=4):
    y_pred_batch = model.predict(X_batch)
    batch_predictions.append(y_pred_batch)
    print(f"Batch predictions: {y_pred_batch}")
```

## Seeding for Reproducibility

All randomized functions support `seed` parameter:

```python
# Reproducible splits
X_train1, y_train1, X_test1, y_test1 = train_test_split(
    X, y, seed=42
)
X_train2, y_train2, X_test2, y_test2 = train_test_split(
    X, y, seed=42
)
# Both splits are identical

# Reproducible batching
for batch1 in batch_iterator(X, y, batch_size=32, seed=42):
    pass

for batch2 in batch_iterator(X, y, batch_size=32, seed=42):
    pass
# Same batch order both times
```

## Complexity Reference

| Function | Time | Space |
|----------|------|-------|
| train_test_split | $O(n)$ | $O(n)$ |
| batch_iterator | $O(n)$ total | $O(b+m)$ per batch |
| shuffle_arrays | $O(n\log n)$ | $O(n)$ |
| add_intercept | $O(nm)$ | $O(nm)$ |

See [Metrics API](metrics.md) for metric complexities and [User Guide](../guide/user_guide.md) for integration examples.

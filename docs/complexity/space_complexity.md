# Space Complexity Analysis

## Overview

This document provides a detailed breakdown of the space (memory) complexity for all operations in OptLinearRegress. Understanding memory usage is crucial for embedded systems and large-scale applications.

## Notation

- $n$ = number of training samples
- $m$ = number of features (including intercept)
- $k$ = number of test samples
- Space measured in number of doubles (8 bytes each)
- $\mathcal{O}(\cdot)$ = Big-O space complexity (asymptotic upper bound)

## Module-by-Module Analysis

## Linear Algebra Module (linalg)

### Matrix Multiplication: py_matmul

Multiplies $A \in \mathbb{R}^{m \times n}$ by $B \in \mathbb{R}^{n \times p}$ to produce $C \in \mathbb{R}^{m \times p}$

**Memory Requirements**:

When called, Python maintains:
- **Input A**: Memoryview (no copy), $\mathcal{O}(1)$ extra
- **Input B**: Memoryview (no copy), $\mathcal{O}(1)$ extra
- **Output C**: Pre-allocated, user responsible, $\mathcal{O}(1)$ extra
- **Temporary variables**: Loop counters and accumulators, $\mathcal{O}(1)$

**Total Space Complexity**: $\mathcal{O}(1)$

**Note**: Caller must allocate C beforehand. Function uses zero-copy memoryviews.

### Matrix-Vector Multiplication: py_matvec

Multiplies $A \in \mathbb{R}^{m \times n}$ by $x \in \mathbb{R}^n$ to produce $y \in \mathbb{R}^m$

**Memory Requirements**:
- **Input A, x**: Memoryviews, $\mathcal{O}(1)$ extra
- **Output y**: Pre-allocated, $\mathcal{O}(1)$ extra
- **Temporaries**: Loop variables, accumulators, $\mathcal{O}(1)$

**Total Space Complexity**: $\mathcal{O}(1)$

### Matrix Transpose: py_transpose

Transposes square matrix $A \in \mathbb{R}^{n \times n}$ in-place

**Memory Requirements**:
- **Input A**: Modified in-place, $\mathcal{O}(1)$ extra
- **Temporaries**: Single swap variable for element exchange, $\mathcal{O}(1)$

**Total Space Complexity**: $\mathcal{O}(1)$

**True in-place operation**: Only temporary variable needed

### Matrix Inversion: py_invert

Inverts square matrix $A \in \mathbb{R}^{n \times n}$ using Gauss-Jordan elimination

**Algorithm**: 
- Augment with identity matrix to form $[A | I]$
- Perform row operations to get $[I | A^{-1}]$

**Memory Requirements**:

```c
double* I = malloc(n * n * sizeof(double))  // Identity matrix
// ... perform operations on both A and I ...
// Copy I back to A
free(I)
```

**Breakdown**:
- **Input A**: Overwritten in-place, $\mathcal{O}(1)$ extra
- **Temporary identity matrix I**: $n \times n$ matrix, $\mathcal{O}(n^2)$
- **Loop variables**: $\mathcal{O}(1)$

**Total Space Complexity**: $\mathcal{O}(n^2)$

**Memory Usage**:
- For $n = 100$ features: $100 \times 100 \times 8$ bytes = 80 KB
- For $n = 1000$ features: $1000 \times 1000 \times 8$ bytes = 8 MB
- For $n = 10000$ features: $10000 \times 10000 \times 8$ bytes = 800 MB

## Solvers Module

### Normal Equation Solver: solve_normal_equation

Solves $(X^T X + \lambda I)\beta = X^T y$ via matrix inversion

**Memory Allocation Breakdown**:

#### Temporary Arrays in Function

```c
double* XtX = malloc(n_features * n_features * sizeof(double))    // Gram matrix
double* Xty = malloc(n_features * sizeof(double))                 // Cross-product
// Invert matrix internally allocates identity: malloc(n_features^2)
```

**Detailed Breakdown**:

1. **Gram Matrix XtX**: $m \times m$
   - Space: $m^2$ doubles = $8m^2$ bytes

2. **Cross-product Xty**: $m$ vector
   - Space: $m$ doubles = $8m$ bytes

3. **Identity Matrix (in invert_matrix)**: $m \times m$
   - Space: $m^2$ doubles = $8m^2$ bytes
   - Allocated temporarily, freed before return

4. **Function parameters (input arrays)**:
   - X: Not allocated by solver (passed as memoryview)
   - y: Not allocated by solver (passed as memoryview)
   - beta: Not allocated by solver (passed as memoryview, filled in-place)

**Maximum Concurrent Memory**:

```
At peak usage (during inversion):
- XtX: m² doubles
- Xty: m doubles
- Identity I: m² doubles
- Input X, y: Not counted (caller responsible)
- Total: 2m² + m ≈ 2m² doubles = 16m² bytes
```

**Total Space Complexity**: $\mathcal{O}(m^2)$

**Memory Usage Examples**:
- For $m = 10$ features: $2 \times 100 \times 8$ bytes ≈ 1.6 KB
- For $m = 100$ features: $2 \times 10,000 \times 8$ bytes = 160 KB
- For $m = 1000$ features: $2 \times 1,000,000 \times 8$ bytes = 16 MB
- For $m = 10000$ features: $2 \times 100,000,000 \times 8$ bytes = 1.6 GB

## Models Module

### LinearRegressor.fit()

Main training method

**Memory Allocation**:

```cython
cdef double* X = malloc(n_samples * n_features * sizeof(double))
cdef double* y = malloc(n_samples * sizeof(double))
cdef double* beta = malloc(n_features * sizeof(double))
```

**Breakdown**:

1. **Feature matrix X** (with intercept prepended):
   - Shape: $n \times m$ (n samples, m features)
   - Space: $nm$ doubles = $8nm$ bytes

2. **Target vector y**:
   - Shape: $(n,)$
   - Space: $n$ doubles = $8n$ bytes

3. **Coefficient vector beta**:
   - Shape: $(m,)$
   - Space: $m$ doubles = $8m$ bytes

4. **Solver temporary arrays** (inside c_solve_normal_equation):
   - Space: $\mathcal{O}(m^2)$ (see solver analysis above)

**Total at peak**:
$$S_{\text{fit}} = nm + n + m + 2m^2 = \mathcal{O}(nm + m^2)$$

**Dominant term**:
- If $n \gg m$: $\mathcal{O}(nm)$ dominates
- If $m$ large: $\mathcal{O}(m^2)$ dominates

**Example Memory Usage**:

| n (samples) | m (features) | nm | m² | Total | Physical |
|---|---|---|---|---|---|
| 1,000 | 10 | 10k | 100 | 10.1k doubles | ~80 KB |
| 10,000 | 50 | 500k | 2.5k | 502.5k doubles | ~4 MB |
| 100,000 | 100 | 10M | 10k | 10.01M doubles | ~80 MB |
| 1,000,000 | 500 | 500M | 250k | 500.25M doubles | ~4 GB |
| 1,000,000 | 1000 | 1B | 1M | 1.001B doubles | ~8 GB |

### LinearRegressor.predict()

Prediction method

**Memory Allocation**:

```cython
cdef double* X = malloc(n_test * n_features * sizeof(double))
cdef double* y_pred = malloc(n_test * sizeof(double))
```

**Breakdown**:

1. **Test feature matrix X** (with intercept):
   - Shape: $k \times m$
   - Space: $km$ doubles = $8km$ bytes

2. **Prediction output y_pred**:
   - Shape: $(k,)$
   - Space: $k$ doubles = $8k$ bytes

**Total Space Complexity**: $\mathcal{O}(km + m)$

Typically: $\mathcal{O}(km)$ if $m$ small

**Note**: No solver operations, so no $m^2$ term!

### LinearRegressor.coefficients()

Getter method

**Memory Allocation**:
```python
return [self.beta[i] for i in range(self.n_features)]
```

**Breakdown**:
- List creation: $m$ elements
- Space: $m$ doubles = $8m$ bytes

**Total Space Complexity**: $\mathcal{O}(m)$

## Utilities Module

### Metrics Functions

#### mean_squared_error

```python
mse = sum((y_true[i] - y_pred[i]) ** 2 for i in range(n)) / n
```

**Memory**:
- Accumulator variable: $\mathcal{O}(1)$
- Input lists: Not allocated by function

**Total Space Complexity**: $\mathcal{O}(1)$

#### r2_score

```python
mean_y_true = sum(y_true) / n                   # scalar
ss_res = sum((y_true[i] - y_pred[i]) ** 2)    # scalar
ss_tot = sum((y_true[i] - mean_y_true) ** 2)  # scalar
```

**Memory**:
- Three accumulator scalars: $\mathcal{O}(1)$
- Input lists: Not allocated

**Total Space Complexity**: $\mathcal{O}(1)$

#### explained_variance_score

```python
diff = [y_true[i] - y_pred[i] for i in range(n)]      # O(n)
mean_diff = sum(diff) / n                              # O(1)
var_res = sum((d - mean_diff) ** 2 for d in diff) / n # O(1)
var_true = sum((y - mean_y_true) ** 2 for y in y_true) / n  # O(1)
```

**Memory**:
- Temporary difference list: $n$ elements = $8n$ bytes
- Accumulator variables: $\mathcal{O}(1)$

**Total Space Complexity**: $\mathcal{O}(n)$

### Data Utilities

#### train_test_split

```python
indices = list(range(n_samples))                    # O(n)
indices.shuffle()                                    # in-place
train_idx = indices[:split_idx]                     # O(n)
test_idx = indices[split_idx:]                      # O(n)
X_train = [X[i] for i in train_idx]                # O(nm) copies
X_test = [X[i] for i in test_idx]                  # O(km) copies
y_train = [y[i] for i in train_idx]                # O(n) copies
y_test = [y[i] for i in test_idx]                  # O(k) copies
```

**Memory Breakdown**:

1. **Index list**: $n$ integers
2. **Output arrays**: 
   - X_train: $(n-k) \times m$ = $\mathcal{O}(nm)$
   - X_test: $k \times m$ = $\mathcal{O}(km)$
   - y_train, y_test: $\mathcal{O}(n)$

**Total Space Complexity**: $\mathcal{O}(nm)$

**Note**: Creates copies of data (necessary for splitting)

#### batch_iterator

Generator-based, yields batches without storing all at once

```python
for start_idx in range(0, n_samples, batch_size):
    batch_idx = indices[start_idx:start_idx + batch_size]  # O(batch_size)
    X_batch = [X[i] for i in batch_idx]                     # O(batch_size * m)
    y_batch = [y[i] for i in batch_idx]                     # O(batch_size)
    yield X_batch, y_batch
```

**Memory for one batch**: $\mathcal{O}(b \cdot m + b) = \mathcal{O}(b \cdot m)$
- Where $b$ = batch_size

**Memory for entire iterator**: $\mathcal{O}(nm)$ if all batches kept in memory
- But generator yields batches, so practical memory: $\mathcal{O}(bm)$

**Total Space Complexity (per batch)**: $\mathcal{O}(bm)$

## Complete End-to-End Memory Profile

### Training Pipeline

```python
from OptLinearRegress.models import LinearRegressor
from OptLinearRegress.utils.data import train_test_split
```

**Step 1: Split data**
- Input: Original dataset (caller responsibility)
- Output: X_train, y_train, X_test, y_test
- Extra memory: $\mathcal{O}(nm)$ for copies

**Step 2: Fit model**
- Input: X_train, y_train (both used as memoryviews)
- Temporary: X matrix, y vector, beta, Gram matrix, identity
- Peak memory: $\mathcal{O}(nm + m^2)$

**Step 3: Predict**
- Input: X_test (used as memoryview)
- Temporary: Augmented test matrix, predictions
- Peak memory: $\mathcal{O}(km)$

**Step 4: Evaluate metrics**
- Input: y_test, y_pred (used as lists/iterables)
- Temporary: Accumulators
- Peak memory: $\mathcal{O}(1)$

### Total Peak Memory

Assuming sequential execution (step 2 after 1, etc.):

$$S_{\text{peak}} = \max(\mathcal{O}(nm), \mathcal{O}(nm + m^2), \mathcal{O}(km), \mathcal{O}(1))$$

$$= \mathcal{O}(nm + m^2)$$

In practice:
- Training data: $\mathcal{O}(nm)$ (caller allocated)
- Model fitting: $\mathcal{O}(nm + m^2)$ (fit function)
- Predictions: $\mathcal{O}(km)$ (can be done in batches)

**Example Memory Usage**:

| Scenario | n | m | k | Training | Prediction | Total |
|----------|---|---|---|----------|-----------|-------|
| Small | 100 | 5 | 20 | ~4K | ~0.8K | ~5KB |
| Medium | 10k | 50 | 2k | ~4MB | ~0.8MB | ~5MB |
| Large | 1M | 100 | 100k | ~800MB | ~80MB | ~900MB |
| Very Large | 1M | 1000 | 100k | ~8GB | ~800MB | ~9GB |

## Memory Optimization Strategies

### 1. Reduce Feature Dimension

Effect: Memory ∝ $m^2$ (quadratic)

```python
# Don't use all features
from sklearn.decomposition import PCA

# Reduce to 20 principal components
pca = PCA(n_components=20)
X_reduced = pca.fit_transform(X_train)
# Now model uses m=20 instead of m=100
# Memory: 100² → 20² = 25x less Gram matrix space
```

### 2. Use Generator-Based Batching

```python
from OptLinearRegress.utils.data import batch_iterator

# Process in batches - do NOT fit all at once
for X_batch, y_batch in batch_iterator(X, y, batch_size=1000):
    # Can't directly fit on batches with current API
    # But predictions work fine per-batch
    y_pred = model.predict(X_batch)
```

### 3. Feature Scaling

```python
# Normalize features (doesn't reduce memory, but improves stability)
def scale_features(X):
    m = [sum(row) / len(row) for row in zip(*X)]
    s = [sum((x[j] - m[j])**2 for x in X)**0.5 / len(X) for j in range(len(X[0]))]
    return [[(x[j] - m[j]) / (s[j]+1e-10) for j in range(len(X[0]))] for x in X]
```

### 4. Prediction in Batches

```python
def predict_in_batches(model, X_test, batch_size=1000):
    predictions = []
    for i in range(0, len(X_test), batch_size):
        X_batch = X_test[i:i+batch_size]
        y_batch = model.predict(X_batch)
        predictions.extend(y_batch)
    return predictions

# Total memory: O(km) → O(bm) where b=batch_size
```

## Comparison with Other Methods

| Method | Fit Memory | Note |
|--------|-----------|------|
| **Normal Equation** | $O(nm + m^2)$ | This implementation |
| **Cholesky** | $O(nm + m^2)$ | Similar, slightly better constant |
| **Gradient Descent** | $O(nm)$ | No matrix inversion, less memory |
| **SGD** | $O(bm)$ | Mini-batch, very memory efficient |
| **SVD** | $O(\min(nm, n^2))$ | Can be higher, more stable |

## Key Takeaways

1. **Gram matrix dominates**: $\mathcal{O}(m^2)$ term in fitting
2. **Feature dimension matters**: Quadratic in features
3. **Prediction is efficient**: Linear in test samples
4. **Metrics are free**: Almost no memory for evaluation
5. **Optimize by**:
   - Reducing features dimension (biggest impact)
   - Predicting in batches
   - Avoiding data copies where possible

## Reference Table

| Component | Space | Notes |
|-----------|-------|-------|
| X (training) | $nm$ | Temporary in fit() |
| y (training) | $n$ | Temporary in fit() |
| Gram matrix | $m^2$ | Main memory concern |
| Beta | $m$ | Stored permanently |
| Predictions | $k$ | Temporary in predict() |
| Metrics | $\leq n$ | Only for explained_var |

See [Time Complexity Analysis](time_complexity.md) for computational complexity and [Summary](summary.md) for quick reference.

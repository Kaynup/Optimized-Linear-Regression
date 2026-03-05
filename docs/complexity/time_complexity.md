# Time Complexity Analysis

## Overview

This document provides a detailed breakdown of the time complexity for all operations in OptLinearRegress, from low-level linear algebra to high-level model operations.

## Notation

- $n$ = number of training samples
- $m$ = number of features (including intercept)
- $k$ = number of test samples
- $\mathcal{O}(\cdot)$ = Big-O time complexity (asymptotic upper bound)

## Module-by-Module Analysis

## Linear Algebra Module (linalg)

### Matrix Multiplication: py_matmul

Computes $C = AB$ where $A \in \mathbb{R}^{m \times n}$, $B \in \mathbb{R}^{n \times p}$, $C \in \mathbb{R}^{m \times p}$

**Algorithm**: Standard triple-nested loop multiplication

```
for i in range(m):              // m iterations
    for j in range(p):          // p iterations
        C[i,j] = 0
        for k in range(n):      // n iterations
            C[i,j] += A[i,k] * B[k,j]
```

**Time Complexity**: $\mathcal{O}(mnp)$

**Breakdown**:
- Outer loop: $m$ iterations
- Middle loop: $p$ iterations per outer iteration
- Inner loop: $n$ iterations per middle iteration
- Each operation: constant time
- Total: $m \times p \times n = \mathcal{O}(mnp)$

**Special Cases**:
- Square matrix: $m = n = p$ → $\mathcal{O}(n^3)$
- Vector-matrix: $p = 1$ → $\mathcal{O}(mn)$

### Matrix-Vector Multiplication: py_matvec

Computes $y = Ax$ where $A \in \mathbb{R}^{m \times n}$, $x \in \mathbb{R}^{n}$, $y \in \mathbb{R}^{m}$

**Algorithm**: Dot product for each row

```
for i in range(m):              // m iterations
    y[i] = 0
    for j in range(n):          // n iterations
        y[i] += A[i,j] * x[j]   // constant time
```

**Time Complexity**: $\mathcal{O}(mn)$

**Breakdown**:
- Outer loop: $m$ iterations
- Inner loop: $n$ iterations per outer iteration
- Each operation: constant time
- Total: $m \times n = \mathcal{O}(mn)$

### Matrix Transpose: py_transpose

Transposes square matrix $A \in \mathbb{R}^{n \times n}$ in-place

**Algorithm**: Swap elements above/below diagonal

```
for i in range(n):              // n iterations
    for j in range(i+1, n):     // ~n/2 iterations per i
        swap(A[i,j], A[j,i])    // constant time
```

**Time Complexity**: $\mathcal{O}(n^2)$

**Breakdown**:
- Outer loop: $n$ iterations
- Inner loop: $\sum_{i=0}^{n-1} (n-1-i) = \frac{n(n-1)}{2} \approx \frac{n^2}{2}$
- Each operation: constant time (swap)
- Total: $\mathcal{O}(n^2)$

**In-Place**: No additional space for data (only swap variable)

### Matrix Inversion: py_invert

Inverts square matrix $A \in \mathbb{R}^{n \times n}$ using Gauss-Jordan elimination

**Algorithm**:

1. **Augment with identity**: Create $[A | I]$ (implicitly)
2. **Forward elimination** (3 nested loops):
   ```
   for i in range(n):              // n pivot iterations
       // normalize pivot row
       for j in range(n):          // n elements
       
       // eliminate column in other rows
       for k in range(n):          // n other rows
           for j in range(n):      // n elements per row
   ```

3. **Back substitution**: (Already done in Jordan phase)
4. **Extract inverse**: Copy auxiliary matrix back

**Time Complexity**: $\mathcal{O}(n^3)$

**Detailed Breakdown**:

Gauss-Jordan elimination pseudocode:
```
for i in range(n):                          // n pivots
    // Step 1: Normalize pivot row
    for j in range(n):                      // n elements: O(n)
        
    // Step 2: Eliminate column in other rows
    for k in range(n):                      // n-1 other rows: ~n
        if k != i:
            for j in range(n):              // n operations per row: O(n)
```

**Complexity analysis**:
- Pivot step: $i = 0$ to $n-1$ (n iterations)
- For each pivot:
  - Normalize: $\mathcal{O}(n)$
  - Eliminate: $(n-1) \times \mathcal{O}(n) = \mathcal{O}(n^2)$
- Total: $n \times \mathcal{O}(n^2) = \mathcal{O}(n^3)$

**Constant Factor**: GJ elimination is $2n^3/3 + \mathcal{O}(n^2)$ flops

**Comparison**:
- Gauss only (to triangular): $n^3/3$ flops
- Gauss-Jordan (to diagonal): $n^3/2$ flops
- Cholesky (for symmetric): $n^3/6$ flops

## Solvers Module

### Normal Equation Solver: solve_normal_equation

Solves $(X^T X + \lambda I) \beta = X^T y$ for $\beta$ via matrix inversion

**Algorithm Steps**:

1. **Compute Gram Matrix**: $G = X^T X + \lambda I$
2. **Compute Cross-Product**: $c = X^T y$
3. **Invert Matrix**: $G^{-1}$
4. **Final Multiplication**: $\beta = G^{-1} c$

**Complexity of Each Step**:

#### Step 1: Gram Matrix Computation ($X^T X$)

Input: $X \in \mathbb{R}^{n \times m}$ (n samples, m features)

```
for i in range(m):                      // m rows of G
    for j in range(m):                  // m columns of G
        G[i,j] = 0
        for k in range(n):              // sum over n samples
            G[i,j] += X[k,i] * X[k,j]  // constant time
        if i == j:
            G[i,i] += alpha             // add regularization
```

**Complexity**: $\mathcal{O}(nm^2)$

**Breakdown**:
- Outer loop: $m$ iterations
- Middle loop: $m$ iterations
- Inner loop: $n$ iterations
- Each operation: constant time
- Total: $m \times m \times n = \mathcal{O}(nm^2)$

**Special Structure**: $G$ is symmetric, so could optimize to $\mathcal{O}(nm^2/2)$ but not done here

#### Step 2: Cross-Product Computation ($X^T y$)

Input: $X \in \mathbb{R}^{n \times m}$, $y \in \mathbb{R}^n$

```
for i in range(m):                  // m features
    c[i] = 0
    for k in range(n):              // n samples
        c[i] += X[k,i] * y[k]       // constant time
```

**Complexity**: $\mathcal{O}(nm)$

**Breakdown**:
- Outer loop: $m$ iterations
- Inner loop: $n$ iterations
- Total: $m \times n = \mathcal{O}(nm)$

#### Step 3: Matrix Inversion

Input: $G \in \mathbb{R}^{m \times m}$ (from Step 1)

(See py_invert analysis above)

**Complexity**: $\mathcal{O}(m^3)$

#### Step 4: Matrix-Vector Multiplication ($\beta = G^{-1} c$)

Input: $G^{-1} \in \mathbb{R}^{m \times m}$, $c \in \mathbb{R}^m$

(See py_matvec analysis above)

**Complexity**: $\mathcal{O}(m^2)$

### Total Solver Complexity

$$T_{\text{solver}} = \mathcal{O}(nm^2) + \mathcal{O}(nm) + \mathcal{O}(m^3) + \mathcal{O}(m^2)$$

$$= \mathcal{O}(nm^2 + m^3)$$

**Which term dominates?**

- If $n \gg m^2$ (very high samples, reasonable features): $\mathcal{O}(nm^2)$ dominates
- If $m$ large (high-dimensional): $\mathcal{O}(m^3)$ dominates
- Typically: Cross-over point at $n \approx m$

**Examples**:

| n | m | $nm^2$ | $m^3$ | Dominant |
|---|---|--------|-------|----------|
| 1000 | 10 | 100,000 | 1,000 | $nm^2$ |
| 10,000 | 10 | 1,000,000 | 1,000 | $nm^2$ |
| 100 | 100 | 1,000,000 | 1,000,000 | Both ($m^3$ slightly) |
| 100 | 500 | 25,000,000 | 125,000,000 | $m^3$ |

## Models Module

### LinearRegressor.fit()

Combines data preparation and solver invocation

**Algorithm**:

1. **Allocate arrays**: $X, y$ (temp buffers) and $\beta$ (output)
2. **Augment features**: Add intercept column to X
3. **Call solver**: invoke c_solve_normal_equation
4. **Deallocate**: Free temporary arrays

**Step-by-Step Complexity**:

#### Step 1: Memory Allocation

```
malloc(n * m * sizeof(double))  // X matrix
malloc(n * sizeof(double))      // y vector
malloc(m * sizeof(double))      // beta vector
```

**Complexity**: $\mathcal{O}(1)$ amortized (allocation time is amortized constant)

#### Step 2: Data Augmentation (Add Intercept)

```
for i in range(n):                          // n samples
    X[i, 0] = 1.0                           // set intercept: O(1)
    for j in range(raw_features):           // m-1 features
        X[i, j+1] = X_py[i][j]              // copy feature: O(1)
    y[i] = y_py[i]                          // copy label: O(1)
```

**Complexity**: $\mathcal{O}(n \times m) = \mathcal{O}(nm)$

**Breakdown**:
- Outer loop: $n$ samples
- Inner loop: $m-1$ features
- Total: $n \times m$

#### Step 3: Solver Call

(See solve_normal_equation analysis above)

**Complexity**: $\mathcal{O}(nm^2 + m^3)$

#### Step 4: Deallocation

**Complexity**: $\mathcal{O}(1)$ amortized

### Total fit() Complexity

$$T_{\text{fit}} = \mathcal{O}(nm) + \mathcal{O}(nm^2 + m^3) = \mathcal{O}(nm^2 + m^3)$$

### LinearRegressor.predict()

Makes predictions on new data

**Algorithm**:

1. **Allocate arrays**: $X_{\text{test}}$ (with intercept), $y_{\text{pred}}$
2. **Augment test features**: Add intercept column
3. **Compute predictions**: Matrix-vector product for each sample
4. **Deallocate**: Free temporary arrays

**Step-by-Step Complexity**:

#### Step 1-2: Augmentation (k test samples)

```
for i in range(k):                          // k test samples
    X[i, 0] = 1.0                           // O(1)
    for j in range(m-1):                    // m-1 features
        X[i, j+1] = X_py[i][j]              // O(1)
```

**Complexity**: $\mathcal{O}(km)$

#### Step 3: Prediction

```
for i in range(k):                          // k test samples
    temp = 0
    for j in range(m):                      // m features (including intercept)
        temp += X[i, j] * beta[j]           // one multiplication: O(1)
    y_pred[i] = temp
```

**Complexity**: $\mathcal{O}(km)$

**Breakdown**:
- Outer loop: $k$ samples
- Inner loop: $m$ features
- Total: $k \times m = \mathcal{O}(km)$

### Total predict() Complexity

$$T_{\text{predict}} = \mathcal{O}(km) + \mathcal{O}(km) = \mathcal{O}(km)$$

**Note**: Linear in both test samples and features - very efficient

### LinearRegressor.coefficients()

Simple getter for learned coefficients

**Algorithm**:

```
return [self.beta[i] for i in range(self.n_features)]
```

**Complexity**: $\mathcal{O}(m)$

**Breakdown**:
- Copy m coefficient values
- Each access: constant time
- Total: $\mathcal{O}(m)$

## Utilities Module

### Metrics Functions

All metrics in utils.metrics operate on predicted and true values:

#### mean_squared_error

```
mse = sum((y_true[i] - y_pred[i])**2 for i in range(n)) / n
```

**Complexity**: $\mathcal{O}(n)$

#### mean_absolute_error

```
mae = sum(abs(y_true[i] - y_pred[i]) for i in range(n)) / n
```

**Complexity**: $\mathcal{O}(n)$

#### r2_score

```
mean_y = sum(y_true) / n                          // O(n)
ss_res = sum((y_true[i] - y_pred[i])**2)        // O(n)
ss_tot = sum((y_true[i] - mean_y)**2)           // O(n)
r2 = 1 - ss_res / ss_tot                        // O(1)
```

**Complexity**: $\mathcal{O}(n)$ (3 linear passes)

**Constant Factor**: 3 passes through data

### Data Splitting Functions

#### train_test_split

```
indices = list(range(n))            // O(n)
if shuffle:
    random.shuffle(indices)         // O(n) Fisher-Yates
split_idx = int(n * (1 - test_size)) // O(1)
train_idx = indices[:split_idx]     // O(n) copying
test_idx = indices[split_idx:]      // O(n) copying
X_train = [X[i] for i in train_idx] // O(n*m)
...
```

**Total Complexity**: $\mathcal{O}(n + nm) = \mathcal{O}(nm)$

**Breakdown**:
- Index creation: $\mathcal{O}(n)$
- Shuffling: $\mathcal{O}(n)$
- Copying data: $\mathcal{O}(nm)$

## End-to-End Complexity

### Complete Training Pipeline

1. **Data splitting**: $\mathcal{O}(nm)$
2. **Model fitting**: $\mathcal{O}(nm^2 + m^3)$ (where $n$ is training samples)
3. **Evaluation predictions**: $\mathcal{O}(km)$ (where $k$ is test samples)
4. **Metrics computation**: $\mathcal{O}(k)$

**Total**: $\mathcal{O}(nm + nm^2 + m^3 + km + k) = \mathcal{O}(nm^2 + m^3 + km)$

In typical scenarios:
- If doing single train-test cycle: $\mathcal{O}(nm^2 + m^3)$
- Dominant term: Usually $m^3$ for reasonable feature dimensions

### Cross-Validation (k-fold)

Single fold: $\mathcal{O}(nm^2 + m^3)$ where $n/k$ is fold size

Total for k folds:
$$T_{\text{CV}} = k \times \mathcal{O}\left(\frac{nm^2}{k} + m^3\right) = \mathcal{O}(nm^2 + km^3)$$

**Dominant term**: $km^3$ (inversion repeated k times)

## Comparison with Alternatives

| Method | Time | Notes |
|--------|------|-------|
| **Normal Equation** | $\mathcal{O}(nm^2 + m^3)$ | Direct solution, no iterations |
| **Gradient Descent** | $\mathcal{O}(nmi)$ | i = iterations, typically slow |
| **SGD** | $\mathcal{O}(bmi)$ | b = batch size, faster but less stable |
| **Cholesky** | $\mathcal{O}(nm^2 + m^3/3)$ | Faster for symmetric matrices |
| **SVD** | $\mathcal{O}(\min(nm^2, n^2m))$ | Most stable, slower |

## Practical Complexity For Common Scenarios

| Scenario | n | m | $nm^2$ | $m^3$ | Total | Time* |
|----------|---|---|--------|-------|-------|-------|
| Small | 100 | 5 | 2.5k | 125 | ~2.6k | ~26 μs |
| Medium | 10k | 20 | 4M | 8k | ~4M | ~4 ms |
| Large | 1M | 100 | 1B | 1M | ~1B | ~1 s |
| HD | 1M | 1000 | 1T | 1B | ~1T | ~1000 s |

*Approximate time on modern CPU (~1B operations/sec)

## Key Takeaways

1. **Fitting scales as**: $\mathcal{O}(nm^2 + m^3)$
   - Prediction: $\mathcal{O}(km)$ (much faster)
   - Evaluation: $\mathcal{O}(k)$

2. **Feature dimension matters more**: $m$ appears with higher powers

3. **Good for**:
   - Small to medium dimensions ($m < 1000$)
   - Many samples relative to features ($n \gg m$)

4. **Poor for**:
   - Very high dimensions ($m > 10000$)
   - $m \approx n$ or $m > n$ (ill-conditioned)

See [Space Complexity Analysis](space_complexity.md) for memory breakdown and [Summary](summary.md) for quick reference tables.

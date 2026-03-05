# Complexity Analysis Summary & Quick Reference

## Quick Complexity Table

### Component Complexities

| Component | Operation | Time | Space | Notes |
|-----------|-----------|------|-------|-------|
| **linalg** | matmul | $O(mnp)$ | $O(1)$ | No temp arrays |
| | matvec | $O(mn)$ | $O(1)$ | Very efficient |
| | transpose | $O(n^2)$ | $O(1)$ | In-place only |
| | invert | $O(n^3)$ | $O(n^2)$ | Gauss-Jordan |
| **solvers** | solve_normal_eq | $O(nm^2 + m^3)$ | $O(m^2)$ | Main bottleneck |
| **models** | fit() | $O(nm^2 + m^3)$ | $O(nm + m^2)$ | Whole solution |
| | predict() | $O(km)$ | $O(km)$ | Very fast |
| | coefficients() | $O(m)$ | $O(m)$ | Simple getter |
| **metrics** | MSE/MAE/RMSE/R² | $O(n)$ | $O(1)$ | All single pass |
| | explaianed_var | $O(n)$ | $O(n)$ | Needs temp array |
| **utils** | train_test_split | $O(nm)$ | $O(nm)$ | Creates copies |
| | batch_iterator | $O(bm)$ | $O(bm)$ | Per batch |

Legend:
- $n$ = samples
- $m$ = features (including intercept)
- $k$ = test samples
- $b$ = batch size

## Dominant Complexity Terms

### Model Fitting: $O(nm^2 + m^3)$

**When does each term dominate?**

```
For typical usage:
- If n >> m²:     O(nm²) dominates (Gram matrix computation)
- If m³ > nm²:    O(m³) dominates (matrix inversion)
- Cross-point at: m ≈ √n
```

**Practical implications**:

| Regime | Characteristic | Dominant Term | Example |
|--------|---|---|---|
| **Many samples** | $n \gg m^2$ | $nm^2$ | 1M samples, 100 features |
| **Balanced** | $n \approx m^2$ | Both | 10k samples, 100 features |
| **High-dimensional** | $m^2 > nm$ | $m^3$ | 1k samples, 1000 features |

### Prediction: $O(km)$

**Prediction is linear in everything** - very efficient compared to fitting

- 1000x better than fitting if good samples-to-features ratio
- Bottleneck only if predicting on very large datasets

## Complexity vs Dataset Size

```
Training Time (theoretical)
↑
│    O(m³) slope
│   /
│  /─────── For high-dimensional
│ /
│/─── O(nm²) slope for small m
└─────────────────────────→
    Number of samples (n)
```

## Memory Scaling Graphs

### Fitting Memory vs Features

```
Memory (GB)
    ↑
   8│         m=1000
    │        ╱
   4│       ╱  m=500
    │      ╱  ╱
   2│     ╱  ╱  m=100
    │    ╱  ╱  ╱
   1│   ╱  ╱  ╱
    │  ╱  ╱  ╱ m=50
  0.5├ ╱  ╱
    │  ▆▆▆▆▆▆▆▆▆▆ (Gram matrix m² dominate)
     └─────────────────→ # features (m)
```

### Fitting Memory vs Samples

```
Memory (MB)
    ↑
 800│              n=1M, m=1000
    │             ╱
 400│            ╱  n=100k, m=1000
    │           ╱  ╱
 200│          ╱  ╱  n=10k
    │         ╱  ╱
 100│        ╱  ╱
    │       ╱  ╱ (X matrix nm dominates)
     └─────────────────→ # samples (n)
```

## Operation Family Complexities

### Linear Algebra Operations

```
matmul    O(mnp)        → General purpose
matvec    O(mn)         → Most common
transpose O(n²)         → Cheap
invert    O(n³)         → EXPENSIVE
```

### Least Squares Solving

```
Gram:      O(nm²)       ← First termin fit
CrossProd: O(nm)        → Negligible
Invert:    O(m³)        ← Second term in fit
Total:     O(nm² + m³)  ← SUM of two terms
```

### Prediction & Evaluation

```
predict()    O(km)      ← Linear
metrics()    O(k)       ← Constant or linear
Together:    O(km)      ← Prediction dominates
```

## Performance Scaling

### Scaling with Samples (n)

Assuming fixed feature dimension (m = 50):

| n | Relative Time | Relative Memory |
|---|---|---|
| 1k | 1x | 1x |
| 10k | 10x | 10x |
| 100k | 100x | 100x |
| 1M | 1000x | 1000x |

**Pattern**: Linear scaling in samples (if m fixed)

### Scaling with Features (m)

Assuming fixed samples (n = 100k):

| m | Relative Time | Relative Memory |
|---|---|---|
| 10 | 1x | 1x |
| 50 | 125x | 25x |
| 100 | 1000x | 100x |
| 500 | 125k x | 2.5k x |

**Pattern**: Cubic scaling in features! (if n fixed)

## Prediction Speedup

**Prediction is $O(km)$ vs Fitting is $O(nm^2 + m^3)$**

```
Speedup ratio = (nm² + m³) / (km)
              = (nm + m²) / k        (if approximate)
              ≈ nm/k    (if m small)
```

**Example**: $n = 100k$, $m = 100$, $k = 1k$

```
Speedup = (100k × 100 + 100²) / (1k × 100)
        ≈ 10M / 100k
        ≈ 100x faster to predict than retrain
```

## Breaking Points & Limits

### When Normal Equation Starts to Struggle

| Criterion | Threshold | Action |
|-----------|-----------|--------|
| **Memory** | $m > 3000$ | Use gradient descent instead |
| **Time** | Training > 1 min | Consider SGD or subsampling |
| **Inversion** | Cond number > $10^8$ | Increase alpha, check features |
| **Dimension** | $m > n$ | Underdetermined, regularize |

### Practical Limits

**With 8GB RAM**:
- Dense matrix limited to ~$\sqrt{1B \text{ elements}} \approx m \leq 11,000$
- But Gram matrix alone: $m^2 \leq 1B$, so $m \leq 31,600$ (theoretical)
- Practical: Keep $m < 5,000$

**With 1GB RAM**:
- Practical limit: $m < 1,000$
- For high-n: $n \leq 100M$ with $m < 10$

## Algorithm Comparison

### Complexity Comparison

```
                  Time              Memory
Normal Equation   O(nm² + m³)       O(nm + m²)  ← This
Gradient Descent  O(nmi)            O(nm)       (i=iterations)
SGD              O(bmi)            O(bm)
Cholesky         O(nm² + m³/3)     O(nm + m²)
SVD              O(min(nm²,n²m))   O(nm)
```

### When to Use Each

| Method | Best For | Limitation |
|--------|----------|-----------|
| **Normal Equation** | $m < 1000$, well-conditioned | Cubic inversion |
| **Cholesky** | Symmetric, small features | Less general |
| **Gradient Descent** | Very high-dimensional | Needs tuning, slow |
| **SGD** | Streaming/online | Less stable |
| **SVD** | High precision needed | Slower but stable |

## Memory Optimization Priority

```
Quick wins (implement first):
1. Reduce features m        → m² improvement
2. Use generator batching   → Linear in batch instead of n
3. Avoid data copies        → Reduce temporary space

Advanced (for extreme cases):
4. Feature selection/PCA    → Cubic in components
5. GPU acceleration         → Different memory model
6. Streaming solvers        → Different algorithm
```

## Key Insights

| Insight | Implication |
|---------|------------|
| $m$ affects as $m^2$ in space | Even small reduction in features helps |
| $m$ affects as $m^3$ in time | Feature selection is critical |
| prediction is $O(km)$ | Inference much cheaper than training |
| Gram matrix is bottleneck | Can't avoid $O(m^2)$ space |
| No $n$ in inversion | Prediction counts don't matter for training |

## Decision Tree

```
Starting: Problem with m features, n samples, k predictions

Do you want to minimize:
├─ TIME?
│  └─ Is m > 1000?
│     ├─ YES: Use SGD or GradDesc (dimension too high)
│     └─ NO: Normal Equation (direct solution fastest)
│
├─ MEMORY?
│  └─ Is m > 5000?
│     ├─ YES: Use SVD or gradient method (reduce m)
│     └─ NO: Normal Equation OK
│
└─ STABILITY?
   └─ Use SVD (most stable)
      or Large regularization alpha
```

## Reference Checklist

- [ ] Which term dominates your complexity? ($nm^2$ vs $m^3$?)
- [ ] Can you reduce features? (biggest impact on both time and memory)
- [ ] Is prediction or training the bottleneck?
- [ ] What's your memory constraint?
- [ ] How stable is your data (condition number)?
- [ ] Is reproducibility/determinism needed?

## Example Scenarios

### Scenario 1: "I have 1M samples, 100 features"

```
Analysis:
- nm² = 1M × 100² = 10B     ← Dominates
- m³ = 100³ = 1M            ← Negligible
- Dominant: O(nm²) ← Time is O(nm²)
- Memory: ~10GB (Gram matrix dominates)
- Action: Subsample data or use SGD
```

### Scenario 2: "I have 10k samples, 1000 features"

```
Analysis:
- nm² = 10k × 1000² = 10B    ← Same as scenario 1 time!
- m³ = 1000³ = 1B            ← Comparable
- Dominant: Mix of both
- Memory: ~16GB (Gram matrix + data)
- Action: Reduce features or use different solver
```

### Scenario 3: "I have 1k samples, 50 features"

```
Analysis:
- nm² = 1k × 50² = 2.5M      ← Small
- m³ = 50³ = 125k            ← Even smaller
- Dominant: Data loading and I/O
- Memory: ~1MB (all matrices)
- Action: Normal Equation is perfect
- Speed: ~milliseconds
```

See [Time Complexity Analysis](time_complexity.md) and [Space Complexity Analysis](space_complexity.md) for detailed breakdowns.

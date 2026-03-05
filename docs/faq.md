# FAQ - Frequently Asked Questions

## General Questions

### Q: Is OptLinearRegress production-ready?

**A**: OptLinearRegress is feature-complete and well-tested for production use in scenarios where:
- You need linear regression (not complex models)
- You want minimal dependencies
- You have embedded system constraints
- You need pure Python at runtime

For large-scale ML applications, consider scikit-learn or PyTorch instead.

### Q: How does OptLinearRegress compare to scikit-learn?

**A**: 

| Aspect | OptLinearRegress | scikit-learn |
|--------|---|---|
| Dependencies | None (runtime) | NumPy, SciPy, etc. |
| Speed | Comparable | Comparable |
| Features | Linear regression only | Hundreds of algorithms |
| API Familiarity | scikit-learn inspired | Standard in ML |
| Embedded systems | Excellent | Not designed for |
| Transparency | Full source visible | Good documentation |

**Use OptLinearRegress if**: Minimal dependencies matter, embedded target, or you need to understand every operation.

**Use scikit-learn if**: You need many algorithms, standard ML pipelines, or extensive features.

### Q: Can I use OptLinearRegress for non-linear regression?

**A**: No. OptLinearRegress strictly implements linear regression. For non-linear relationships, you can:

1. **Feature engineering**: Create polynomial/interaction features
   ```python
   X_poly = [[x[0], x[0]**2, x[0]*x[1], ...] for x in X]
   model.fit(X_poly, y)
   ```

2. **Use different library**: Try scikit-learn, TensorFlow, or PyTorch for non-linear methods

### Q: What is the performance compared to alternatives?

**A**: OptLinearRegress aims for comparable performance to NumPy-based scikit-learn for small-to-medium problems. For very large features, other methods may be better.

See [Performance Benchmarks](performance.md) for detailed comparisons.

---

## Technical Questions

### Q: What data types does OptLinearRegress accept?

**A**: Only `list[list[float]]` for features and `list[float]` for targets.

**NOT supported**:
- NumPy arrays
- Pandas DataFrames
- Tuples
- Integer lists

**Solution**:
```python
import numpy as np

# Convert NumPy to list
X_list = X_array.tolist()

# Convert Pandas to list
X_list = df_features.values.tolist()
```

### Q: How do I handle categorical features?

**A**: Use one-hot encoding before passing to the model:

```python
def one_hot_encode(value, categories):
    """Convert categorical value to one-hot vector."""
    return [1.0 if v == value else 0.0 for v in categories]

# Example: color feature
colors = ['red', 'green', 'blue']
encoded_color = one_hot_encode('red', colors)  # [1, 0, 0]
```

### Q: Can I use OptLinearRegress with missing data?

**A**: No. Handle missing data before training:

```python
# Method 1: Remove samples with missing values
X_clean = [X[i] for i in range(len(X)) if None not in X[i] and 'nan' not in str(X[i])]

# Method 2: Impute with mean
def impute_features(X):
    n_features = len(X[0])
    means = [sum(row[j] for row in X) / len(X) for j in range(n_features)]
    return [[X[i][j] if X[i][j] is not None else means[j] for j in range(n_features)] 
            for i in range(len(X))]

X_imputed = impute_features(X)
```

### Q: What should my alpha (regularization) parameter be?

**A**: 

1. **Start with default**: `alpha=1e-8` (minimal regularization)
2. **If singular matrix error**: Increase to `1e-6` or `1e-4`
3. **For regularization effect**: Try `0.01` to `1.0`
4. **Cross-validate** for best value:
   ```python
   # See User Guide for cross-validation code
   ```

**Guidelines**:
- Alpha too small: Numerical instability, singular matrix
- Alpha too large: Over-regularization, poor fit
- No universal best value: Depends on your data

### Q: How do I interpret the learned coefficients?

**A**: 
```python
coeffs = model.coefficients()
print(f"Intercept (β₀): {coeffs[0]}")      # y-intercept
print(f"Feature 1 (β₁): {coeffs[1]}")      # Change per unit of feature 1
print(f"Feature 2 (β₂): {coeffs[2]}")      # Change per unit of feature 2
```

**Interpretation example** (house prices):
```
Model: price = 50000 + 100*sqft - 5000*distance
  - Intercept: 50000 (base price)
  - Sqft coefficient: 100 (price increases $100/sqft)
  - Distance coefficient: -5000 (price decreases $5000/mile)
```

**Note**: Coefficients only meaningful for **normalized features**. Otherwise, scale depends on feature units.

### Q: Should I normalize my features?

**A**: **Yes, strongly recommended**. Benefits:

1. **Easier interpretation** (coefficients in standard units)
2. **Better numerical stability** (Gram matrix better conditioned)
3. **Faster convergence** (regularization more effective)

```python
def normalize(X):
    means = [sum(col)/len(col) for col in zip(*X)]
    stds = [sum((x-m)**2 for x in col)**0.5/len(col) 
            for col, m in zip(zip(*X), means)]
    return [[(x[j]-means[j])/stds[j] for j in range(len(x))] 
            for x in X]

X_norm = normalize(X)
```

### Q: Can I use categorical targets?

**A**: No. Linear regression requires continuous numerical targets.

For classification:
- Use scikit-learn's `LogisticRegression`
- Use other classifiers (SVM, Random Forest, Neural Networks)

---

## Performance & Optimization

### Q: Why is my model slow?

**A**: Check the complexity:

1. **Feature dimension (m)**: Time scales as $O(m^3)$ - biggest factor
   - Solution: Reduce features via PCA or feature selection

2. **Sample count (n)**: Time scales as $O(nm^2)$
   - Solution: Subsample training data

3. **Matrix operations**: Dominated by Gram matrix computation
   - Solution: Can't avoid for Normal Equation, use SGD for huge m

See [Complexity Analysis](../complexity/summary.md) for details.

### Q: How do I handle very large datasets?

**A**: Use sampling or different algorithm:

```python
# Option 1: Random subsample
import random
sample_size = 100000
indices = random.sample(range(len(X)), sample_size)
X_sample = [X[i] for i in indices]
y_sample = [y[i] for i in indices]

model.fit(X_sample, y_sample)

# Option 2: Batch predictions (still memory-efficient)
from OptLinearRegress.utils.data import batch_iterator
for X_batch, _ in batch_iterator(X_test, [0]*len(X_test), batch_size=10000):
    y_pred_batch = model.predict(X_batch)
    # Process batch results
```

### Q: I'm getting memory errors. What can I do?

**A**: 

1. **Reduce features** (quadratic benefit):
   ```python
   X_reduced = [[x[i] for i in important_features] for x in X]
   ```

2. **Reduce samples**:
   ```python
   X_small = X[:100000]
   y_small = y[:100000]
   ```

3. **Use server with more RAM** (obvious but valid)

See [Memory Optimization](../guide/memory_optimization.md) for details.

### Q: How long should training take?

**A**: Depends on your problem size:

| n (samples) | m (features) | ~Time |
|---|---|---|
| 1k | 10 | < 1 ms |
| 100k | 50 | ~10 ms |
| 1M | 100 | ~1 second |
| 10M | 100 | ~10 seconds |
| 1M | 500 | ~100-200 seconds |

Times are on modern CPU (2-3 GHz). Using faster CPU or GPU can improve 10-100x.

---

## Troubleshooting

### Q: "Matrix not invertible" error - what do I do?

**A**: The Gram matrix is singular. Solutions:

1. **Increase alpha (regularization)**:
   ```python
   model = LinearRegressor(alpha=1e-4)  # Was 1e-8
   ```

2. **Check for duplicate features**:
   ```python
   # Remove linearly dependent columns
   ```

3. **Check for constant features**:
   ```python
   # Features with zero variance cause singularity
   varying_features = [j for j in range(m) if any(x[j] != X[0][j] for x in X)]
   X_clean = [[x[j] for j in varying_features] for x in X]
   ```

### Q: R² is negative. What does that mean?

**A**: Model performs worse than just predicting the mean. This means:

1. **Features are irrelevant** to the target
2. **Model is overfit** (overfitting on noise)
3. **Data quality issues** (outliers, errors)

Solutions:
- Check feature relevance
- Validate data quality
- Increase regularization (alpha)
- Use different features

### Q: 我的模型过拟合了。怎么办?

**A**: (Chinese example) Model overfits (training R² >> test R²). Solutions:

1. **Increase regularization**:
   ```python
   model = LinearRegressor(alpha=0.1)  # Stronger penalty
   ```

2. **Use more training data**:
   ```python
   # Get more samples, don't just use same data
   ```

3. **Reduce features**:
   ```python
   # Use only most important features
   ```

4. **Check train/test split**:
   ```python
   # Ensure proper random split with seed
   X_train, y_train, X_test, y_test = train_test_split(
       X, y, seed=42
   )
   ```

### Q: Predictions are always around the mean value

**A**: Model learned almost nothing. Causes:

1. **Features are irrelevant**: No relationship with target
2. **Alpha too large**: Over-regularization killed the signal
3. **Scale issues**: Features in very different scales

Solutions:
- Normalize features
- Reduce alpha
- Verify feature relevance
- Check data quality

---

## Development & Contributing

### Q: Can I modify a fitted model?

**A**: Not directly. Create a new model instance:

```python
# Wrong - don't modify alpha after fitting
model.alpha = 0.5  # This doesn't re-fit!

# Right - create new model
model_new = LinearRegressor(alpha=0.5)
model_new.fit(X_train, y_train)
```

### Q: How do I contribute?

**A**: See `CONTRIBUTING.md` in the repository. We welcome:
- Bug reports
- Performance improvements
- Documentation improvements
- Test additions

### Q: Is the source code documented?

**A**: Yes. Check:
- Docstrings in Python (.pyx files)
- Comments in C++ (.cpp files)
- This documentation site
- In-code complexity analysis

### Q: Can I use this in commercial products?

**A**: Check the LICENSE file. Most open-source licenses allow commercial use with attribution.

---

## Mathematical Questions

### Q: Why does linear regression use sum of squares?

**A**: 

1. **Mathematical**: Directly leads to closed-form solution (Normal Equation)
2. **Statistical**: Maximum likelihood under Gaussian error assumption
3. **Computational**: Convex (unique minimum), no local optima
4. **Physical**: Energy minimization principle

See [Algorithm Overview](../guide/algorithms.md) for more details.

### Q: What's the difference between fitting and prediction?

**A**: 

**Fitting** (learning):
- Optimizes parameters to explain training data
- Solves: $(X^T X + \lambda I)\beta = X^T y$
- Time: $O(nm^2 + m^3)$

**Prediction** (inference):
- Applies learned parameters to new data
- Computes: $\hat{y} = X\beta$
- Time: $O(km)$ much faster!

### Q: Why add a regularization term (alpha)?

**A**: 

1. **Numerical stability**: Prevents singular matrix inversion
2. **Statistical**: Reduces overfitting (bias-variance tradeoff)
3. **Practical**: Makes solution more robust
4. **Theory**: Ridge regression property

Formula: $(X^T X + \lambda I)$ vs just $(X^T X)$

### Q: What is the intercept term?

**A**: The intercept (bias) is automatically added:

```
Given features [x₁, x₂, ..., xₘ]
Model uses features [1, x₁, x₂, ..., xₘ]
                     ↑ automatic intercept column
```

This shifts the prediction line to fit the data better.

---

## Getting Help

### Where to find answers:

1. **This FAQ** - Check here first for common issues
2. **API Documentation** - [docs/api/](../api/models.md)
3. **User Guide** - [docs/guide/user_guide.md](user_guide.md)
4. **Complexity Analysis** - [docs/complexity/](../complexity/summary.md)
5. **Source Code** - Comments are detailed and educational

### Still stuck?

- Check the [User Guide](user_guide.md) for detailed examples
- Review [Best Practices](best_practices.md) for optimization
- Examine test files in `tests/` for usage examples
- Consult [References](../references.md) for theory

---

**Last Updated**: March 2026

Have a question not answered here? Check the source code comments or open an issue in the repository.

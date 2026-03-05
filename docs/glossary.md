# Glossary

## Mathematical Terms

### Algorithm
A step-by-step procedure for solving a problem or performing a computation. In OptLinearRegress, the Normal Equation is the primary algorithm.

### Asymptotic Analysis
Mathematical analysis of algorithm complexity as input size approaches infinity. Expressed using Big-O notation. See [Complexity Analysis](complexity/summary.md).

### Bias
1. **Statistical bias**: Difference between expected value of estimator and true value
2. **Intercept term**: The constant term in linear equation $y = \beta_0 + \beta_1 x_1 + ...$
   - OptLinearRegress automatically includes this as the first coefficient

### Big-O Notation
Mathematical notation describing worst-case time/space complexity. Examples:
- $\mathcal{O}(n)$ - Linear time in n
- $\mathcal{O}(n^2)$ - Quadratic time
- $\mathcal{O}(n^3)$ - Cubic time

See [Complexity Summary](complexity/summary.md).

### Cholesky Decomposition
Factorization of symmetric positive-definite matrix: $A = L L^T$. More efficient than general matrix inversion for special matrices.

### Coefficient (Parameter)
Learned parameter in linear model: $y = \beta_0 + \sum_{j=1}^m \beta_j x_j$
- $\beta_0$ is intercept
- $\beta_1, ..., \beta_m$ are feature coefficients

Returned by `model.coefficients()`.

### Condition Number
Measure of numerical stability of a matrix. High condition number (ill-conditioned) means:
- Small input changes cause large output changes
- Numerical errors amplified
- Solution: Increase regularization (alpha)

### Convex Optimization
Mathematical optimization where objective function is convex (single global minimum). Least squares is convex, so Normal Equation finds the true optimum.

### Cross-Product Vector
Result of $X^T y$ computation in Normal Equation. Vector of size m containing dot products of each feature with targets.

### Design Matrix
Feature matrix X with samples as rows and features as columns. OptLinearRegress automatically adds intercept column, making it $(n \times (m+1))$.

### Eigenvalue / Eigenvector
Scalar $\lambda$ and vector $v$ such that $Av = \lambda v$. Used in SVD and spectral analysis.

---

## Algorithm Terms

### Gaussian Elimination
Systematic method for solving linear systems via row operations. Produces upper triangular matrix.

### Gauss-Jordan Elimination
Extension of Gaussian elimination that produces diagonal matrix (and directly computes inverse). Used by `py_invert()`.

Steps:
1. Forward elimination (Gaussian)
2. Back substitution (Jordan phase)
3. Extract inverse from augmented matrix

Time complexity: $\mathcal{O}(n^3)$

### Gram Matrix
Result of $X^T X$ computation. Square $(m \times m)$ matrix where element $(i,j)$ is dot product of features $i$ and $j$ across all samples.

Used in Normal Equation to solve: $(X^T X + \lambda I)\beta = X^T y$

### Ill-Conditioned Matrix
Matrix where small perturbations cause large changes in solution. Symptoms:
- High condition number
- Singular or near-singular
- Solution: Increase regularization (alpha)

### Linear System
Equation of form $A\beta = b$ where A is matrix, $\beta$ and $b$ are vectors. Normal Equation is a linear system.

### Matrix Inversion
Computing $A^{-1}$ such that $A A^{-1} = I$. Not all matrices are invertible:
- Singular matrix: No inverse (det = 0)
- Near-singular: Inverse very large (numerically unstable)

Implemented via Gauss-Jordan elimination with $\mathcal{O}(n^3)$ complexity.

### Normal Equation
Closed-form solution to least squares problem:
$$\beta^* = (X^T X + \lambda I)^{-1} X^T y$$

Advantages: Exact solution, no iterations. Disadvantage: $\mathcal{O}(n^3)$ inversion.

### Pivot
In Gaussian elimination, the non-zero element used to eliminate others in that column. Pivot selection is crucial for numerical stability.

### Rank Deficiency
When matrix has rank less than number of columns, meaning columns are linearly dependent. Causes singular matrix problems.

### Residual
Difference between predicted and actual values: $r_i = y_i - \hat{y}_i$

Used in:
- Residual sum of squares: $SS_{res} = \sum r_i^2$
- Error analysis and diagnostic plots

---

## Statistical Terms

### Bias-Variance Tradeoff
Fundamental tradeoff in machine learning:
- **Bias**: Error from model assumptions
- **Variance**: Sensitivity to training data
- Regularization (alpha) adds bias to reduce variance

### Coefficient of Determination (R²)
Proportion of variance explained by model:
$$R^2 = 1 - \frac{SS_{res}}{SS_{tot}}$$
- Range: $(-\infty, 1]$
- $R^2 = 1$: Perfect predictions
- $R^2 = 0$: No better than mean
- $R^2 < 0$: Worse than mean

### Degrees of Freedom
Number of independent values in computation. In regression:
- Degrees of freedom = n (samples) - m (parameters)
- Used in statistical testing and confidence intervals

### Explained Variance Score
Similar to R² but based on variance rather than sum of squares. Detects bias in residuals.

### Generalization Error
Prediction error on unseen data. Good model has:
- Low training error
- Low generalization error
- They're close (not overfit)

### Least Squares
Optimization criterion: minimize $\sum_i (y_i - \hat{y}_i)^2$

Normal Equation is the closed-form solution.

### Maximum Likelihood Estimation (MLE)
Statistical principle: choose parameters that maximize probability of observing data.

For linear regression with Gaussian errors, MLE = Least Squares.

### Overfitting
When model fits training data too well, including noise:
- High training accuracy
- Low test accuracy
- Reduces generalization

Solutions: Regularization (alpha), reduce features, more data

### Regularization
Adding penalty term to prevent overfitting:
$$\min ||y - X\beta||^2 + \lambda ||\beta||_2^2$$

L2 regularization (Ridge): Penalty proportional to coefficient magnitude

Parameter "alpha" in OptLinearRegress controls $\lambda$.

### Sum of Squares
- **SS_res** (residual): $\sum_i (y_i - \hat{y}_i)^2$ - unexplained variation
- **SS_tot** (total): $\sum_i (y_i - \bar{y})^2$ - total variation
- Used in R² computation and ANOVA

---

## Computational Terms

### Array Layout
How multidimensional arrays are stored in linear memory:
- **Row-major (C order)**: Used by OptLinearRegress
  - Element [i,j] at position i*ncols + j
  - Faster for row operations
- **Column-major (Fortran order)**: Alternative
  - Element [i,j] at position j*nrows + i
  - Faster for column operations

### Cython
Python-like language that compiles to C. OptLinearRegress uses Cython to:
- Write fast code in Python-like syntax
- Call C/C++ functions
- Use memoryviews for zero-copy operations

### C++ / C++17
Systems programming language with modern features. OptLinearRegress uses C++17 for:
- Linear algebra kernels
- Dynamic memory management
- Performance-critical operations

### Cache Efficiency
How well algorithm uses CPU cache:
- **Good**: Sequential memory access, small working set
- **Bad**: Random access, large arrays
- OptLinearRegress optimizes for cache via row-major layout

### Floating-Point
Approximate representation of real numbers (IEEE 754 standard):
- Double precision: 64-bit, ~15 decimal digits
- Used throughout OptLinearRegress
- Introduces rounding errors

### Memory Allocation
Reserving heap memory for arrays. OptLinearRegress uses:
- `malloc()` for dynamic allocation
- `free()` for deallocation
- Careful management to prevent leaks

### Memoryview
Cython feature providing zero-copy view of array data. Allows passing arrays without copying.

### Row-Major Order
See "Array Layout" above. Default in OptLinearRegress.

### Stack vs Heap
- **Stack**: Fast, limited size, automatic cleanup
- **Heap**: Slower, large capacity, manual management
- Matrices stored on heap via malloc()

---

## Data Science Terms

### Batch Iterator
Generator yielding data in chunks. Useful for:
- Memory efficiency
- Processing large datasets
- Parallel processing

OptLinearRegress provides `batch_iterator()` utility.

### Cross-Validation
Technique for model evaluation:
- Split data into k folds
- Train on k-1, test on 1
- Repeat k times
- Average results

Reduces variance in performance estimate.

### Feature
Input variable to the model, also called:
- Predictor
- Independent variable
- Dimension
- Attribute

OptLinearRegress requires numerical features.

### Feature Engineering
Creating new features from existing ones:
- Polynomial features: $x, x^2, x^3, ...$
- Interaction terms: $x_1 x_2, x_1 x_3, ...$
- Domain-specific transformations

### Feature Scaling
Normalization to consistent scale:
- Mean 0: $(x - \mu) / \sigma$
- Range [0,1]: $(x - \min) / (\max - \min)$

Improves numerical stability and regularization.

### Hyperparameter
Parameter set before training, then not changed:
- Alpha (regularization strength)
- Test set size
- Random seed

Distinguished from "parameters" which are learned.

### Intercept
Constant term in model: $y = \beta_0 + \sum \beta_j x_j$
- $\beta_0$ is intercept
- OptLinearRegress includes automatically

### Label
Target variable to predict, also called:
- Target
- Dependent variable
- Ground truth

### Metric
Measure of model performance:
- MSE: Mean squared error
- R²: Coefficient of determination
- MAE: Mean absolute error

OptLinearRegress provides several metrics.

### Normalization
Scaling features to standardized scale. Benefits:
- Comparable magnitudes
- Better numerical stability
- More effective regularization

### Prediction
Output of model on new data: $\hat{y} = X\beta$

Distinguished from "training" which learns parameters.

### Target (Label)
Value to predict for each sample. Must be numerical for regression.

### Test Set
Data held out during training for evaluation. Used to estimate generalization error.

### Train-Test Split
Dividing data into training (for fitting) and testing (for evaluation) sets.

OptLinearRegress provides `train_test_split()` utility.

### Training Set
Data used to fit the model parameters.

---

## Performance Terms

### Time Complexity
asymptotic running time as function of input size. Examples:
- $\mathcal{O}(n)$ - Linear time
- $\mathcal{O}(nm^2)$ - Quadratic in m
- $\mathcal{O}(n^3)$ - Cubic time

### Space Complexity
Asymptotic memory usage as function of input size. Examples:
- $\mathcal{O}(1)$ - Constant (no extra memory)
- $\mathcal{O}(n)$ - Linear in samples
- $\mathcal{O}(m^2)$ - Quadratic in features

### Throughput
Number of operations completed per unit time. Higher is better.

### Latency
Time to complete single operation. Lower is better.

---

## Abbreviations

| Abbreviation | Meaning |
|---|---|
| **API** | Application Programming Interface |
| **CSV** | Comma-Separated Values |
| **GJ** | Gauss-Jordan |
| **MAE** | Mean Absolute Error |
| **ML** | Machine Learning |
| **MSE** | Mean Squared Error |
| **OLS** | Ordinary Least Squares |
| **QR** | QR Decomposition |
| **RMSE** | Root Mean Squared Error |
| **SGD** | Stochastic Gradient Descent |
| **SVD** | Singular Value Decomposition |
| **SS** | Sum of Squares |

---

## Index by Category

### Linear Algebra
- Algorithm, Coefficient, Eigenvalue, Eigenvector, Gaussian Elimination, Gauss-Jordan Elimination, Gram Matrix, Ill-Conditioned Matrix, Inversion, Pivot, Rank Deficiency, Singular Matrix

### Statistics
- Bias, Coefficient of Determination, Degrees of Freedom, Generalization Error, Maximum Likelihood, Overfitting, Regularization, Sum of Squares

### Optimization
- Convex Optimization, Least Squares, Normal Equation, Ridge Regression

### Data Science
- Batch Iterator, Cross-Validation, Feature, Feature Engineering, Hyperparameter, Label, Normalization, Prediction, Target, Test Set, Training Set

### Computing
- Array Layout, Cython, C++, Cache Efficiency, Floating-Point, Memory Allocation, Memoryview, Stack vs Heap

### Performance
- Space Complexity, Time Complexity, Throughput, Latency

---

See [References](references.md) for academic papers on these concepts.

For API definitions, see [API Reference](api/models.md).

**Last Updated**: March 2026

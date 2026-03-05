# Installation & Quick Start

## System Requirements

- **Python**: 3.8 or higher
- **C++ Compiler**: C++17 compatible (GCC 7+, Clang 5+, MSVC 19.1+)
- **Build Tools**: pip, setuptools
- **Dependencies**: Cython >= 3.0 (for building)

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd Optimized-Linear-Regression
```

### 2. Install Build Dependencies

```bash
pip install -r requirements.txt
```

This installs Cython and other build dependencies.

### 3. Build Cython Extensions

```bash
python setup.py build_ext --inplace
```

This builds the C++ extensions in-place, making them immediately available for import.

### 4. (Optional) Install as Package

```bash
pip install .
```

Or for development mode:

```bash
pip install -e .
```

## 5-Minute Quick Start

### Basic Usage

```python
from OptLinearRegress.models import LinearRegressor
from OptLinearRegress.utils.metrics import mean_squared_error, r2_score

# Prepare training data
X_train = [
    [1.0, 2.0],
    [2.0, 3.0],
    [3.0, 4.0],
    [4.0, 5.0]
]
y_train = [3.0, 5.0, 7.0, 9.0]

# Create and fit model
model = LinearRegressor(alpha=1e-8)
coefficients = model.fit(X_train, y_train)
print("Learned coefficients (including intercept):", coefficients)

# Make predictions
y_pred = model.predict(X_train)
print("Predictions:", y_pred)

# Evaluate
mse = mean_squared_error(y_train, y_pred)
r2 = r2_score(y_train, y_pred)
print(f"MSE: {mse:.6f}")
print(f"R² Score: {r2:.6f}")
```

### With Regularization

```python
# Ridge regression with L2 regularization
model = LinearRegressor(alpha=0.1)  # L2 penalty parameter
model.fit(X_train, y_train)
```

The `alpha` parameter controls the L2 regularization strength:
- `alpha=1e-8` (default): Almost no regularization
- `alpha=0.1`: Moderate regularization
- `alpha=1.0`: Strong regularization

## Running Tests

```bash
# Build extensions and run tests
python setup.py build_ext --inplace && python -m pytest tests/

# Or run tests with verbose output
python -m pytest tests/ -v
```

## Cleaning Up Build Artifacts

```bash
python setup.py clean --all
rm -rf build/
```

## Troubleshooting

### Build Issues

**Issue**: C++17 compiler not found
- **Solution**: Ensure you have a compatible C++ compiler installed
  - Linux: `sudo apt-get install build-essential`
  - macOS: `xcode-select --install`
  - Windows: Install Microsoft Visual Studio Build Tools

**Issue**: "Cython not found"
- **Solution**: Install Cython: `pip install Cython>=3.0`

**Issue**: Module import fails
- **Solution**: Ensure extensions are built: `python setup.py build_ext --inplace`

### Runtime Issues

**Issue**: "Cannot use LinearRegressor - not compiled"
- **Solution**: Rebuild extensions and ensure the build succeeded

**Issue**: Memory errors on large datasets
- **Solution**: See [Memory Optimization](memory_optimization.md) for guidance

## Next Steps

- Read the [User Guide](user_guide.md) for comprehensive examples
- Check [API Reference](../api/models.md) for detailed documentation
- Review [Time Complexity Analysis](../complexity/time_complexity.md) for performance characteristics
- See [Best Practices](best_practices.md) for optimization tips

## Important Notes

1. **Data Format**: Input data must be lists of lists (2D arrays), not NumPy arrays or other formats
2. **Intercept**: The model automatically adds a bias term (intercept) to features
3. **Numerical Stability**: The regularization parameter `alpha` helps with matrix inversion stability
4. **Python Version**: Ensure you're using Python 3.8+ for compatibility

For more information, see the [User Guide](user_guide.md) or visit the [API Reference](../api/models.md).

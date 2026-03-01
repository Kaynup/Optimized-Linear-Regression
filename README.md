# OptLinearRegress

A modular **Cython-based package for Linear Regression** built from scratch. It is designed to be installed on embedded systems, time-critical environments, and systems that require highly optimized and fast calculations without the overhead of heavy dependencies like NumPy.

## 🚀 Features
- **High Performance:** Core logic implemented in Cython and C++ (C++17 standard) for maximum performance.
- **Modularity:** Divided cleanly into `linalg`, `models`, `solvers`, and `utils`.
- **Low Dependencies:** Only requires `Cython >= 3.0` to build the C extensions.
- **Regularization:** Built-in support for Ridge Regression via the `alpha` hyperparameter.
- **Comprehensive Metrics:** Includes mean squared error, mean absolute error, root mean squared error, r2 score, and explained variance.

## 📦 Installation

Requirements:
- Python 3.8+
- C++17 compatible compiler

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd Optimized-Linear-Regression
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Build the Cython extensions in-place:**
   ```bash
   python setup.py build_ext --inplace
   ```

4. **Install the package (Optional):**
   ```bash
   python setup.py install
   ```

## 💡 Quick Start Usage

Here is a quick example of how to use the `LinearRegressor` and evaluate it using built-in metrics:

```python
from OptLinearRegress.models import LinearRegressor
from OptLinearRegress.utils.metrics import mean_squared_error, r2_score

# 1. Prepare data (lists of identical shape)
X_train = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
y_train = [1.5, 2.5, 3.5]

# 2. Initialize the model with an optional L2 regularization term (alpha)
model = LinearRegressor(alpha=1e-8)

# 3. Fit the model to the data
model.fit(X_train, y_train)

# View the learned coefficients (including intercept)
print("Coefficients:", model.coefficients())

# 4. Make predictions
y_pred = model.predict(X_train)

# 5. Evaluate the model
mse = mean_squared_error(y_train, y_pred)
r2 = r2_score(y_train, y_pred)

print(f"Mean Squared Error: {mse:.4f}")
print(f"R² Score: {r2:.4f}")
```

## 🧪 Testing

To ensure everything is working correctly, you can run the test suite with `pytest`:
```bash
# Build extensions and run tests
python setup.py build_ext --inplace && python -m pytest tests/

# Or run tests directly if extensions are already built
python -m pytest tests/
```

To clean up build artifacts:
```bash
python setup.py clean --all
```

## 📁 File Hierarchy

```text
OptLinearRegress
├── __init__.py
├── linalg/           # Linear algebra operations (matrix multiplication, transpose, etc.)
├── models/           # Machine learning classes like LinearRegressor
├── solvers/          # Core solvers (e.g., normal equation calculation)
└── utils/            # Data management, Cython memoryviews, and metrics
```

## 🤝 Contributing
See `CONTRIBUTING.md` for guidelines on how to contributing.

## 📄 License
This project is licensed. See the `LICENSE` file for details.
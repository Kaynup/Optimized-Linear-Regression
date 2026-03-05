# OptLinearRegress Documentation

Welcome to the **OptLinearRegress** documentation. This is a high-performance, Cython-based linear regression library designed for embedded systems, time-critical environments, and applications requiring minimal dependencies.

## --| Documentation Structure

### 1. **Getting Started & User Guide**
- [Installation & Quick Start](guide/quickstart.md) - Setup instructions and basic usage
- [User Guide](guide/user_guide.md) - Comprehensive tutorial and examples
- [Data Handling](guide/data_handling.md) - Data preparation and utilities

### 2. **API Reference**
- [Models API](api/models.md) - LinearRegressor class and methods
- [Solvers API](api/solvers.md) - Normal Equation solver documentation
- [Linear Algebra API](api/linalg.md) - Matrix operations and utilities
- [Metrics API](api/metrics.md) - Performance evaluation functions
- [Utils API](api/utils.md) - Helper functions and data utilities

### 3. **Complexity Analysis**
- [Time Complexity Analysis](complexity/time_complexity.md) - Detailed time complexity breakdown
- [Space Complexity Analysis](complexity/space_complexity.md) - Memory usage analysis
- [Algorithm Complexity Summary](complexity/summary.md) - Quick reference table

### 4. **Algorithms & Theory**
- [Linear Regression Algorithm](guide/algorithms.md) - Mathematical foundations
- [Normal Equation Solver](guide/normal_equation.md) - Least squares solution method
- [Regularization (Ridge Regression)](guide/regularization.md) - L2 regularization theory

### 5. **Performance & Optimization**
- [Performance Benchmarks](guide/performance.md) - Speed comparisons with other libraries
- [Memory Optimization](guide/memory_optimization.md) - Memory efficiency details
- [Best Practices](guide/best_practices.md) - Optimization tips for your use case

### 6. **References & Resources**
- [References & Citations](references.md) - Academic papers and resources
- [Glossary](glossary.md) - Definitions of key terms
- [FAQ](faq.md) - Frequently asked questions

## | Key Features

- **High Performance**: Core algorithms implemented in Cython (Python) and C++17
- **Low Dependencies**: Only requires Cython for compilation, pure Python at runtime
- **Modular Design**: Clean separation between models, solvers, and linear algebra
- **Memory Efficient**: Designed for embedded systems and low-memory environments
- **Regularization Support**: Built-in L2 regularization (Ridge Regression)
- **Comprehensive Metrics**: MSE, MAE, RMSE, R², explained variance

## >> Quick Navigation

**Want to...**

- **Get started?** → [Installation & Quick Start](guide/quickstart.md)
- **Learn the algorithms?** → [Linear Regression Algorithm](guide/algorithms.md)
- **Check API details?** → [API Reference Overview](api/models.md)
- **Understand performance?** → [Time Complexity Analysis](complexity/time_complexity.md)
- **Optimize for your case?** → [Best Practices](guide/best_practices.md)
- **Find code examples?** → [User Guide](guide/user_guide.md)

## Module Architecture

```
OptLinearRegress/
├── linalg/          → Matrix operations (multiplication, transpose, inversion)
├── models/          → Machine learning model (LinearRegressor)
├── solvers/         → Linear system solvers (Normal Equation)
└── utils/
    ├── metrics.py   → Evaluation metrics (MSE, R², etc.)
    └── data.py      → Data utilities (train_test_split, batching)
```

## Version Information

- **Current Version**: 1.0
- **Python Support**: 3.8+
- **C++ Standard**: C++17
- **Build System**: Cython 3.0+

## Contributing & License

Please refer to `CONTRIBUTING.md` for contribution guidelines. This project is licensed under the license specified in `LICENSE`.

---

**Last Updated**: March 2026

For more details, start with [Installation & Quick Start](guide/quickstart.md) or browse the specific topics above.

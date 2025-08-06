# 🤝 Contributing to OptLinearRegress

We welcome contributions to **OptLinearRegress**, especially from those interested in performance engineering, numerical methods, or Cython/C++ systems!

---

## What You Can Contribute

- **New Features**
  - Ridge/Lasso regularization
  - Batch predictions or batched matrix operations
  - Memory-mapped data loading for large datasets

- **Performance Enhancements**
  - Multi-threaded matrix ops with OpenMP or parallel loops
  - BLAS/LAPACK backend integration (optional)

- **Documentation**
  - API usage examples and notebooks
  - Performance benchmarks or comparisons

- **Testing**
  - Unit tests using `pytest`
  - Edge-case handling (e.g., singular matrices)

---

## 🛠 Development Setup

```bash
git clone https://github.com/Kaynup/Optimized-Linear-Regression.git
cd OptLinearRegress
pip install -r requirements.txt
python setup.py build_ext --inplace
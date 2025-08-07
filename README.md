# ⚡️ OptLinearRegress

**OptLinearRegress** is a minimal, high-performance implementation of linear regression written in **Cython (C++)**, designed for scenarios with **<10 features** and **millions of samples**. It is optimized to outperform `scikit-learn`'s `LinearRegression` in specific, performance-critical workloads.

---

## Features

- Optimized for **low-dimensional, large-sample** datasets  
- Simple API: `.fit()`, `.predict()`, `.coefficients()`
- Uses **C-level memory** and **Cython speed**  
- Includes optional **Ridge regularization** for numerical stability  
- Comes with a **benchmark script** against `scikit-learn`

---

## Benchmark (8 features, 5 million samples)

| Model              | Fit Time |
|-------------------|----------|
| OptLinearRegress  | ~150 ms  |
| scikit-learn      | ~250 ms  |

Approx. **100ms speedup** → ~**40% improvement**

---

## 📦 Installation

```bash
git clone https://github.com/yourusername/OptLinearRegress.git
cd OptLinearRegress
pip install -r requirements.txt
python setup.py build_ext --inplace
```

## 📫 Contact

For questions, suggestions, or collaborations:

- Email: punyak.dei@gmail.com  
- GitHub: [@Kaynup](https://github.com/Kaynup)

Please file issues or feature requests on [GitHub Issues](https://github.com/Kaynup/Optimized-Linear-Regression/issues).

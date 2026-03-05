# References & Citations

## Foundational Papers

### Linear Regression & Least Squares

1. **Legendre, A. M. (1805).** "Nouvelles méthodes pour la détermination des orbites des comètes." 
   - Introduction of the least squares method
   - Foundational work for linear regression theory

2. **Gauss, C. F. (1809).** "Theoria Motus Corporum Coelestium."
   - Independent development of least squares
   - Applications to celestial mechanics

### Normal Equations & Matrix Inversion

3. **Golub, G. H., & Van Loan, C. F. (2013).** "Matrix Computations (4th ed.)." Johns Hopkins University Press.
   - Comprehensive reference on numerical linear algebra
   - Detailed analysis of Gauss-Jordan elimination
   - Conditioning and stability of matrix operations

4. **Strang, G. (2009).** "Introduction to Linear Algebra (4th ed.)." Wellesley-Cambridge University Press.
   - Clear exposition of least squares problems
   - Normal equations derivation and interpretation

### Regularization

5. **Tikhonov, A. N. (1943).** "On the stability of inverse problems."
   - Theory of regularization
   - Ill-posed problems and solutions

6. **Hoerl, A. E., & Kennard, R. W. (1970).** "Ridge Regression: Biased Estimation for Nonorthogonal Problems." 
   - *Technometrics*, 12(1), 55-67.
   - Ridge regression and L2 regularization theory first application to regression

### Computational Complexity

7. **Demmel, J. W. (1997).** "Applied Numerical Linear Algebra." SIAM.
   - Computational complexity analysis of linear algebra algorithms
   - Floating-point error analysis

## Numerical Methods

### Gaussian Elimination & Variants

8. **Trefethen, L. N., & Bau III, D. (1997).** "Numerical Linear Algebra." SIAM.
   - Comprehensive treatment of Gaussian elimination
   - Gauss-Jordan elimination (Lecture 22)
   - Computational cost analysis

9. **Press, W. H., Teukolsky, S. A., Vetterling, W. T., & Flannery, B. P. (2007).** "Numerical Recipes: The Art of Scientific Computing (3rd ed.)." Cambridge University Press.
   - Practical implementations of numerical methods
   - Linear system solving (Chapter 2, "Solution of Linear Algebraic Equations")

### Alternative Methods

10. **Cholesky, A. L. (1910).** "Method of solving certain integral equations with applications to boundary value problems."
    - Cholesky decomposition
    - More efficient for symmetric positive definite matrices

11. **Golub, G. H., & Pereyra, V. (1973).** "The differentiation of pseudo-inverses and nonlinear least-squares problems whose variables separate."
    - SVD for least squares
    - Numerical stability in solution

## Implementation & Optimization

### Cython & Python Performance

12. **Behnel, S., Bradshaw, R., Citro, C., Dalcin, L., Seljebotn, D. S., & Smith, K. (2011).** "Cython: The Best of Both Worlds." 
    - *Computing in Science & Engineering*, 13(2), 31-39.
    - Performance optimization through Cython

13. **Van Rossum, G., & Drake Jr, F. L. (2009).** "The Python Language Reference Manual."
    - Python language and performance considerations

### C++ for Scientific Computing

14. **Stroustrup, B. (2013).** "The C++ Programming Language (4th ed.)." Addison-Wesley.
    - C++11/14/17 features for scientific computing
    - Memory management and performance

## Mathematical References

### Linear Algebra Foundations

15. **Horn, R. A., & Johnson, C. R. (2012).** "Matrix Analysis (2nd ed.)." Cambridge University Press.
    - Comprehensive linear algebra theory
    - Matrix norms and condition numbers

16. **Boyd, S., & Vandenberghe, L. (2004).** "Convex Optimization." Cambridge University Press.
    - Optimization theory underlying linear regression
    - Least squares as a special case of convex optimization

### Statistics & Machine Learning

17. **Hastie, T., Tibshirani, R., & Friedman, J. (2009).** "The Elements of Statistical Learning: Data Mining, Inference, and Prediction (2nd ed.)." Springer.
    - Chapter 3: "Linear Methods for Regression"
    - Bias-variance tradeoff and regularization

18. **Bishop, C. M. (2006).** "Pattern Recognition and Machine Learning." Springer.
    - Chapter 3: "Linear Models for Regression"
    - Bayesian perspective on linear regression

19. **Wasserman, L. (2004).** "All of Statistics: A Concise Course in Statistical Inference." Springer.
    - Statistical foundations
    - Inference and hypothesis testing

## Software Engineering References

### API Design

20. **Scikit-learn Development Team.** "scikit-learn: Machine Learning in Python."
    - *Journal of Machine Learning Research*, 12, 2825-2830 (2011).
    - Our API design is inspired by scikit-learn's interface
    - Consistency and usability principles

21. **McKinney, W. (2010).** "Data Structures for Statistical Computing in Python."
    - *Proceedings of the 9th Python in Science Conference*, 51-56.
    - Best practices for scientific Python libraries

## Complexity Theory

### Big-O Notation & Analysis

22. **Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2009).** "Introduction to Algorithms (3rd ed.)." MIT Press.
    - Chapter 3: "Growth of Functions"
    - Asymptotic notation and complexity analysis

23. **Knuth, D. E. (1997).** "The Art of Computer Programming, Volume 1: Fundamental Algorithms (3rd ed.)." Addison-Wesley.
    - Analysis of algorithms
    - Numeric methods

## Stability & Conditioning

24. **Kahan, W. (1966).** "Numerical linear algebra."
    - Pioneering work on numerical stability
    - Floating-point error analysis

25. **Skeel, R. D. (1980).** "Scaling for Numerical Stability in Gaussian Elimination."
    - *Journal of the ACM*, 26(3), 494-526.
    - Condition numbers and numerical stability

## Online Resources

### Documentation & Tutorials

- **NumPy Documentation**: https://numpy.org/doc/
  - Reference for array operations (we provide list interfaces)
  
- **SciPy Linear Algebra**: https://docs.scipy.org/doc/scipy/reference/linalg.html
  - Advanced matrix operations

- **Scikit-learn Documentation**: https://scikit-learn.org/
  - API design inspiration
  - Comparison baseline

### Educational

- **MIT OpenCourseWare - Linear Algebra**: https://ocw.mit.edu/courses/mathematics/18-06-linear-algebra-spring-2010/
  - Professor Gilbert Strang's lectures
  
- **Stanford CS109 - Data Science**: https://cs109.github.io/
  - Machine learning and statistics foundations

## Cited Concepts & Techniques

### In Order of Appearance

| Concept | Primary References | Section |
|---------|-------------------|---------|
| Least Squares | Legendre (1805), Gauss (1809) | Algorithms |
| Normal Equations | Strang (2009), Golub & Van Loan (2013) | Solvers |
| Gaussian Elimination | Trefethen & Bau (1997), Press et al. (2007) | Linear Algebra |
| Gauss-Jordan Elimination | Golub & Van Loan (2013) | Linear Algebra |
| Ridge Regression | Tikhonov (1943), Hoerl & Kennard (1970) | Regularization |
| Cholesky Decomposition | Cholesky (1910), Golub & Van Loan (2013) | Alternatives |
| SVD | Golub & Pereyra (1973) | Alternatives |
| Condition Numbers | Kahan (1966), Horn & Johnson (2012) | Stability |
| Floating-Point Errors | Kahan (1966), Demmel (1997) | Implementation |
| Big-O Analysis | Cormen et al. (2009), Knuth (1997) | Complexity |

## How to Cite OptLinearRegress

For academic use, please cite:

```bibtex
@software{optlinearregress2024,
  title={OptLinearRegress: Optimized Linear Regression for Embedded Systems},
  author={Your Research Team},
  year={2024},
  url={https://github.com/your-repo/Optimized-Linear-Regression}
}
```

## Related Software

### Similar Libraries

1. **scikit-learn**: Comprehensive ML library (higher-level, more dependencies)
2. **NumPy/SciPy**: Fundamental numerical computing (lower-level)
3. **Statsmodels**: Statistical modeling (more comprehensive statistics)
4. **JAX**: Composable transformations of NumPy functions (modern alternative)
5. **PyTorch**: Deep learning framework (for neural networks approach)

### For Comparison

- **LAPACK**: Foundational linear algebra library (Fortran)
- **BLAS**: Basic Linear Algebra Subprograms (Fortran optimization)
- **Intel MKL**: High-performance math kernel library
- **Eigen**: C++ template library for linear algebra

## Further Reading Roadmap

### Beginner

1. Start with: Strang (2009) - "Introduction to Linear Algebra"
2. Then: Hastie et al. (2009) - Chapter 3 for statistical perspective

### Intermediate

1. Golub & Van Loan (2013) - Matrix Computations for numerical methods
2. Boyd & Vandenberghe (2004) - Convex Optimization for theory

### Advanced

1. Kahan (1966) and Demmel (1997) - Numerical stability deep dive
2. Horn & Johnson (2012) - Advanced linear algebra theory
3. Cormen et al. (2009) - Complexity analysis foundations

## Acknowledgments

OptLinearRegress design and API are inspired by the excellent scikit-learn library. We thank the developers of Cython, Python, and C++ for providing the tools that make this project possible.

---

**Last Updated**: March 2026

For the most current references and citations, see the source code comments and docstrings throughout OptLinearRegress.

# 🤝 Contributing to OptLinearRegress

Thank you for your interest! Whether you’re optimizing performance or improving documentation—every contribution helps.

**Table of Contents**
- [What You Can Contribute](#what-you-can-contribute)
- [How to Get Started](#how-to-get-started)
- [Issue & PR Workflow](#issue--pr-workflow)
- [Testing & CI](#testing--ci)
- [Need Help?](#need-help)
- [Code of Conduct](#code-of-conduct)
- [A Note of Appreciation](#a-note-of-appreciation)

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

## How to Get Started
Clone, set up dev environment—link to README for more details.

```bash
git clone https://github.com/Kaynup/Optimized-Linear-Regression.git
cd OptLinearRegress
pip install -r requirements.txt
python setup.py build_ext --inplace
```

## Issue & PR Workflow
1. Open an issue to discuss your proposal.
2. Once agreed, fork the repo, implement your feature, and open a PR.
3. Use [PR checklist template] for clarity.

## Testing & CI
Run `pytest` (include benchmarks if any). CI builds validate tests for every PR.

## Need Help?
Comment on issues or ping me directly—happy to guide you!

## Code of Conduct
Please follow our [Code of Conduct](link-to-CODE_OF_CONDUCT.md).

## A Note of Appreciation
Thank you for helping make OptLinearRegress better!

# Linear Algebra API Reference

## Overview

The `linalg` module provides low-level linear algebra operations implemented in Cython with C/C++ backends for maximum performance.

## Core Operations

### Matrix Multiplication

**Function**: `py_matmul`

Multiplies two matrices: $C = AB$ where $A \in \mathbb{R}^{m \times n}$ and $B \in \mathbb{R}^{n \times p}$.

**Signature**:
```python
def py_matmul(A_flat: memoryview[double],
              B_flat: memoryview[double],
              C_flat: memoryview[double],
              m: int,
              n: int,
              p: int) -> None:
```

**Parameters**:
- `A_flat`: Flattened matrix A (memoryview)
  - Shape: `(m*n,)`
  - Row-major layout
  
- `B_flat`: Flattened matrix B (memoryview)
  - Shape: `(n*p,)`
  - Row-major layout
  
- `C_flat`: Output matrix C (memoryview, pre-allocated)
  - Shape: `(m*p,)`
  - Row-major layout
  
- `m`: Rows of A (and C)
- `n`: Columns of A, rows of B
- `p`: Columns of B (and C)

**Time Complexity**: $O(mnp)$
- Standard triple nested loop implementation
- No algorithmic optimization (e.g., Strassen) for simplicity

**Space Complexity**: $O(1)$
- Only uses input/output arrays (no temporary allocations)

**Algorithm**:
```
for i in range(m):
    for j in range(p):
        C[i*p + j] = 0
        for k in range(n):
            C[i*p + j] += A[i*n + k] * B[k*p + j]
```

**Example**:
```python
import array
from OptLinearRegress.linalg.linalg import py_matmul

# Multiply [2x3] x [3x2] -> [2x2]
A_flat = array.array('d', [1, 2, 3, 4, 5, 6])  # [[1,2,3], [4,5,6]]
B_flat = array.array('d', [1, 2, 3, 4, 5, 6])  # [[1,2], [3,4], [5,6]]
C_flat = array.array('d', [0, 0, 0, 0])        # output [2x2]

py_matmul(A_flat, B_flat, C_flat, m=2, n=3, p=2)
# C_flat is now [22, 28, 49, 64]
```

### Matrix-Vector Multiplication

**Function**: `py_matvec`

Multiplies a matrix by a vector: $y = Ax$ where $A \in \mathbb{R}^{m \times n}$ and $x \in \mathbb{R}^n$.

**Signature**:
```python
def py_matvec(A_flat: memoryview[double],
              x_flat: memoryview[double],
              y_flat: memoryview[double],
              m: int,
              n: int) -> None:
```

**Parameters**:
- `A_flat`: Flattened matrix (memoryview)
  - Shape: `(m*n,)`
  - Row-major layout
  
- `x_flat`: Input vector (memoryview)
  - Shape: `(n,)`
  
- `y_flat`: Output vector (memoryview, pre-allocated)
  - Shape: `(m,)`
  
- `m`: Rows of A
- `n`: Columns of A, length of x

**Time Complexity**: $O(mn)$
- Single pass through matrix and vector

**Space Complexity**: $O(1)$
- No temporary allocations

**Algorithm**:
```
for i in range(m):
    y[i] = 0
    for j in range(n):
        y[i] += A[i*n + j] * x[j]
```

**Example**:
```python
import array
from OptLinearRegress.linalg.linalg import py_matvec

# Multiply [3x2] x [2] -> [3]
A_flat = array.array('d', [1, 2, 3, 4, 5, 6])  # [[1,2], [3,4], [5,6]]
x_flat = array.array('d', [1, 2])
y_flat = array.array('d', [0, 0, 0])             # output [3]

py_matvec(A_flat, x_flat, y_flat, m=3, n=2)
# y_flat is now [5, 11, 17]
```

### Matrix Transpose

**Function**: `py_transpose`

Transposes a square matrix in-place: $A \leftarrow A^T$.

**Signature**:
```python
def py_transpose(A_flat: memoryview[double],
                 n: int) -> None:
```

**Parameters**:
- `A_flat`: Flattened square matrix (memoryview)
  - Shape: `(n*n,)`
  - Row-major layout
  
- `n`: Dimension of square matrix

**Time Complexity**: $O(n^2)$
- Must visit each element once

**Space Complexity**: $O(1)$
- In-place operation (single temporary variable for swaps)

**Algorithm**:
```
for i in range(n):
    for j in range(i+1, n):
        swap(A[i*n + j], A[j*n + i])
```

**Example**:
```python
import array
from OptLinearRegress.linalg.linalg import py_transpose

# Transpose [3x3] in-place
A_flat = array.array('d', [1, 2, 3, 4, 5, 6, 7, 8, 9])
# [[1, 2, 3],
#  [4, 5, 6],
#  [7, 8, 9]]

py_transpose(A_flat, n=3)
# A_flat is now [1, 4, 7, 2, 5, 8, 3, 6, 9]
# [[1, 4, 7],
#  [2, 5, 8],
#  [3, 6, 9]]
```

### Matrix Inversion

**Function**: `py_invert`

Inverts a square matrix in-place using Gauss-Jordan elimination: $A \leftarrow A^{-1}$.

**Signature**:
```python
def py_invert(A_flat: memoryview[double],
              n: int) -> int:
```

**Parameters**:
- `A_flat`: Flattened square matrix (memoryview)
  - Shape: `(n*n,)`
  - Row-major layout
  - Will be overwritten with inverse
  
- `n`: Dimension of matrix

**Returns**:
- `0`: Success
- `-1`: Singular matrix (non-invertible)
- `-2`: Memory allocation failure

**Time Complexity**: $O(n^3)$
- Gauss-Jordan elimination: $O(n^3)$ operations

**Space Complexity**: $O(n^2)$
- Temporary identity matrix for computation

**Algorithm**:

1. **Create augmented matrix** $[A | I]$
   
2. **Forward elimination** (Gauss):
   - For each pivot column $i$:
     - Find pivot row (non-zero element)
     - Normalize pivot row
     - Eliminate column below pivot

3. **Back substitution** (Jordan):
   - For each pivot row:
     - Eliminate column above pivot

4. **Result**: $[I | A^{-1}]$ (inverse replaces original)

**Pseudo-code**:
```
for i in range(n):
    # Find pivot and eliminate
    pivot = A[i*n + i]
    if pivot == 0:
        return -1  # singular
    
    # Normalize row
    for j in range(n):
        A[i*n + j] /= pivot
        I[i*n + j] /= pivot
    
    # Eliminate other rows
    for k in range(n):
        if k != i:
            temp = A[k*n + i]
            for j in range(n):
                A[k*n + j] -= temp * A[i*n + j]
                I[k*n + j] -= temp * I[i*n + j]

# Copy inverse back to A
for i in range(n):
    for j in range(n):
        A[i*n + j] = I[i*n + j]
```

**Example**:
```python
import array
from OptLinearRegress.linalg.linalg import py_invert

# Invert a [2x2] matrix
A_flat = array.array('d', [4.0, 7.0, 2.0, 6.0])  # [[4, 7], [2, 6]]
status = py_invert(A_flat, n=2)

if status == 0:
    print("Inverse:", A_flat)
    # A_flat is now [0.6, -0.7, -0.2, 0.4]
    # [[0.6, -0.7], [-0.2, 0.4]]
elif status == -1:
    print("Matrix is singular")
```

## Stability Considerations

### Numerical Stability

1. **GJ Elimination**: Generally stable for well-conditioned matrices
2. **Pivot Selection**: Implementation uses partial pivoting implicitly
3. **Regularization**: Always use with regularization term (alpha)

### Ill-Conditioned Matrices

For matrices with high condition numbers:
- Use L2 regularization (alpha parameter)
- Consider feature scaling
- Check for linearly dependent columns

## Memory Layout

All functions use **row-major (C-contiguous)** layout:

```
For a [3x2] matrix:
     [[1, 2],
      [3, 4],
      [5, 6]]

Flattened: [1, 2, 3, 4, 5, 6]
Access: A[i*ncols + j]
```

## Performance Tips

1. **Minimize Allocations**: Pre-allocate output arrays
2. **Use In-Place Operations**: Where possible
3. **Correct Layout**: Ensure row-major layout
4. **Avoid Copies**: Pass memoryviews for zero-copy access

## C-Level Functions

Direct C functions are available for:
- `matmul_c(A, B, C, m, n, p)`
- `matvec_c(A, x, y, m, n)`
- `transpose_inplace(A, n)`
- `invert_matrix(A, n)`

These are called by Python wrappers but not directly accessible.

## Complexity Summary Table

| Operation | Time | Space | In-Place |
|-----------|------|-------|----------|
| Matmul | $O(mnp)$ | $O(1)$ | No |
| Matvec | $O(mn)$ | $O(1)$ | No |
| Transpose | $O(n^2)$ | $O(1)$ | Yes |
| Invert | $O(n^3)$ | $O(n^2)$ | Yes (overwrites) |

See [Solvers API](solvers.md) for higher-level usage and [Complexity Analysis](../complexity/time_complexity.md) for detailed breakdown.

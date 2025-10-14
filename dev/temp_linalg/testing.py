import time
from array import array
import linalg

# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------
def make_matrix(n, start=1.0):
    """Create a simple n×n test matrix as array('d')."""
    return array("d", [float(i + start) for i in range(n * n)])

def make_vector(n, start=1.0):
    """Create a simple n-element vector."""
    return array("d", [float(i + start) for i in range(n)])

def as_2d_view(a, n):
    """Cast 1D double array → 2D double memoryview (n×n)."""
    return memoryview(a).cast("B").cast("d", (n, n))

def as_1d_view(a):
    """Cast 1D double array → 1D double memoryview."""
    return memoryview(a).cast("B").cast("d")

def arr_to_list2d(a, n):
    """Convert 1D array('d') to nested list."""
    return [[a[i * n + j] for j in range(n)] for i in range(n)]

# ------------------------------------------------------------
# Python reference implementations
# ------------------------------------------------------------
def py_matmul(A, B, n):
    C = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            s = 0.0
            for k in range(n):
                s += A[i][k] * B[k][j]
            C[i][j] = s
    return C

def py_matvec(A, x, n):
    y = [0.0] * n
    for i in range(n):
        s = 0.0
        for j in range(n):
            s += A[i][j] * x[j]
        y[i] = s
    return y

def py_transpose(A, n):
    return [[A[j][i] for j in range(n)] for i in range(n)]

def py_invert_2x2(A):
    """Reference invert for 2x2 only."""
    det = A[0][0]*A[1][1] - A[0][1]*A[1][0]
    if det == 0:
        return None
    inv_det = 1.0 / det
    return [[A[1][1]*inv_det, -A[0][1]*inv_det],
            [-A[1][0]*inv_det, A[0][0]*inv_det]]

# ------------------------------------------------------------
# 1. Matrix multiplication test
# ------------------------------------------------------------
print("=== Matrix Multiply Test ===")
N = 4
A = make_matrix(N)
B = make_matrix(N, start=2.0)
C = array("d", [0.0] * (N * N))

mvA = as_2d_view(A, N)
mvB = as_2d_view(B, N)
mvC = as_2d_view(C, N)

t0 = time.perf_counter()
linalg.matmul(mvA, mvB, mvC)
t1 = time.perf_counter()
cy_time = (t1 - t0) * 1e6

pyA = arr_to_list2d(A, N)
pyB = arr_to_list2d(B, N)
t0 = time.perf_counter()
pyC = py_matmul(pyA, pyB, N)
t1 = time.perf_counter()
py_time = (t1 - t0) * 1e6

diff = max(abs(mvC[i, j] - pyC[i][j]) for i in range(N) for j in range(N))

print(f"Size: {N}x{N}")
print(f"Cython time: {cy_time:.2f} µs | Python time: {py_time:.2f} µs | Speedup: {py_time/cy_time:.1f}x")
print(f"Max diff: {diff:.3e}")

# ------------------------------------------------------------
# 2. Matrix-vector test
# ------------------------------------------------------------
print("\n=== Matrix-Vector Test ===")
x = make_vector(N)
y = array("d", [0.0] * N)
mvx = as_1d_view(x)
mvy = as_1d_view(y)

t0 = time.perf_counter()
linalg.matvec(mvA, mvx, mvy)
t1 = time.perf_counter()
cy_time = (t1 - t0) * 1e6

pyx = [float(i + 1) for i in range(N)]
t0 = time.perf_counter()
pyy = py_matvec(pyA, pyx, N)
t1 = time.perf_counter()
py_time = (t1 - t0) * 1e6

diff = max(abs(mvy[i] - pyy[i]) for i in range(N))

print(f"Size: {N}")
print(f"Cython time: {cy_time:.2f} µs | Python time: {py_time:.2f} µs | Speedup: {py_time/cy_time:.1f}x")
print(f"Max diff: {diff:.3e}")

# ------------------------------------------------------------
# 3. Transpose test
# ------------------------------------------------------------
print("\n=== Transpose Test ===")
N = 3
A = make_matrix(N)
mvA = as_2d_view(A, N)

pyA = arr_to_list2d(A, N)
pyT = py_transpose(pyA, N)

t0 = time.perf_counter()
linalg.transpose(mvA)
t1 = time.perf_counter()
cy_time = (t1 - t0) * 1e6

diff = max(abs(mvA[i, j] - pyT[i][j]) for i in range(N) for j in range(N))
print(f"Size: {N}x{N}")
print(f"Cython time: {cy_time:.2f} µs | Max diff: {diff:.3e}")

# ------------------------------------------------------------
# 4. Inversion test (2×2 case)
# ------------------------------------------------------------
print("\n=== Inversion Test (2x2) ===")
A = array("d", [4.0, 7.0, 2.0, 6.0])
mvA = as_2d_view(A, 2)
pyA = [[4.0, 7.0], [2.0, 6.0]]

t0 = time.perf_counter()
res = linalg.invert(mvA)
t1 = time.perf_counter()
cy_time = (t1 - t0) * 1e6

py_inv = py_invert_2x2(pyA)

if res == 0 and py_inv is not None:
    diff = max(abs(mvA[i, j] - py_inv[i][j]) for i in range(2) for j in range(2))
    print(f"Inversion success, Cython time: {cy_time:.2f} µs | Max diff: {diff:.3e}")
else:
    print("Matrix is singular or inversion failed.")

import numpy as np
import time
import matplotlib.pyplot as plt

# -----------------------------
# Gram–Schmidt QR (your version)
# -----------------------------
def gram_schmidt_qr(A):
    n, m = A.shape
    if n != m:
        raise ValueError(f"the matrix A is not square, {A.shape=}")

    Q = np.empty_like(A)
    R = np.zeros_like(A)

    for j in range(n):
        u = A[:, j].copy()

        for i in range(j):
            R[i, j] = np.dot(Q[:, i], A[:, j])
            u -= R[i, j] * Q[:, i]

        R[j, j] = np.linalg.norm(u)
        Q[:, j] = u / R[j, j]

    return Q, R


# -----------------------------
# LU factorisation (your version)
# -----------------------------
def lu_factorisation(A):
    n, m = A.shape
    if n != m:
        raise ValueError(f"Matrix A is not square {A.shape}")

    L = np.zeros((n, n), dtype=float)
    U = np.zeros((n, n), dtype=float)

    for i in range(n):
        L[i, i] = 1.0

    for j in range(n):

        # U row
        for i in range(j + 1):
            s = 0.0
            for k in range(i):
                s += L[i, k] * U[k, j]

            U[i, j] = A[i, j] - s

        if U[j, j] == 0:
            raise ValueError("Zero pivot, LU without pivoting fails.")

        # L column
        for i in range(j + 1, n):
            s2 = 0.0
            for k in range(j):
                s2 += L[i, k] * U[k, j]

            L[i, j] = (A[i, j] - s2) / U[j, j]

    return L, U


# -----------------------------
# Matrix A
# -----------------------------
A = np.array([
    [4, 2, 0],
    [2, 3, 1],
    [0, 1, 2.5]
], dtype=float)


# -----------------------------
# Runtime test function
# -----------------------------
def time_function(func, A, repeats=20000):
    times = []

    for _ in range(repeats):
        start = time.perf_counter()
        func(A)
        end = time.perf_counter()
        times.append(end - start)

    return np.array(times)


# -----------------------------
# Measure run times
# -----------------------------
qr_times = time_function(gram_schmidt_qr, A)
lu_times = time_function(lu_factorisation, A)

print("QR times: min =", qr_times.min(),
      "max =", qr_times.max(),
      "mean =", qr_times.mean())

print("LU times: min =", lu_times.min(),
      "max =", lu_times.max(),
      "mean =", lu_times.mean())


# -----------------------------
# Plot run times
# -----------------------------
plt.figure(figsize=(10,5))
plt.plot(qr_times, label="Gram-Schmidt QR")
plt.plot(lu_times, label="LU Factorisation")
plt.xlabel("Run number")
plt.ylabel("Time (seconds)")
plt.title("Runtime Comparison: QR vs LU")
plt.legend()
plt.grid(True)
out_path = "runtimes.png"
plt.savefig(out_path)
plt.close()
print(f"Saved plot to {out_path}")
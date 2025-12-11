import numpy as np 
def lu_factorisation(A):
    """
    Compute the LU factorisation of a square matrix A.

    The function decomposes a square matrix ``A`` into the product of a lower
    triangular matrix ``L`` and an upper triangular matrix ``U`` such that:

        A = L U

    where ``L`` has unit diagonal elements and ``U`` is upper triangular.

    Parameters
    ----------
    A : numpy.ndarray
        A 2D NumPy array of shape ``(n, n)`` representing the square matrix to
        factorise.

    Returns
    -------
    L : numpy.ndarray
        A lower triangular matrix with shape ``(n, n)`` and unit diagonal.
    U : numpy.ndarray
        An upper triangular matrix with shape ``(n, n)``.
    """

    n, m = A.shape
    if n != m:
        raise ValueError(f"Matrix A is not square {A.shape}")

    # Create empty L and U
    L = np.zeros((n, n), dtype=float)
    U = np.zeros((n, n), dtype=float) 

    # put 1s on the diagonal of L
    for i in range(n):
        L[i, i] = 1.0
 
    for j in range(n):

        # compute U entries
        for i in range(j + 1):

            s = 0.0
            for k in range(i):
                s = s + (L[i, k] * U[k, j])

            # U formula 
            U[i, j] = A[i, j] - s

        # pivot check 
        if U[j, j] == 0:
            raise ValueError("Zero pivot, you can't do LU without pivoting).")

        # compute L entries
        for i in range(j + 1, n):

            s2 = 0.0
            for k in range(j):
                s2 = s2 + (L[i, k] * U[k, j])

            # divide by diagonal of U
            num = A[i, j] - s2
            L[i, j] = num / U[j, j]

    return L, U
    
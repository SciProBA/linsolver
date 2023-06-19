"""Routines for solving a linear system of equations."""
import numpy as np

# Tolerance value for dependency check
_DEPENDENCY_TOL = 1e-10


def gaussian_eliminate(coeffs, rhs):
    """Solves a linear system of equations (Ax = b) by Gauss-elimination

    Args:
        coeffs: Matrix with the coefficients. Shape: (n, n).
        rhs: Right hand side of the equation. Shape: (n,)

    Returns:
        Vector xx with the solution of the linear equation.

    Raises:
        ValueError: if the system of equation is linearly dependent.
    """
    nn = coeffs.shape[0]
    for ii in range(nn):
        imax = np.argmax(np.abs(coeffs[ii:, ii])) + ii
        if imax != ii:
            coeffs[ii], coeffs[imax] = np.array(coeffs[imax]), np.array(coeffs[ii])
            rhs[ii], rhs[imax] = np.array(rhs[imax]), np.array(rhs[ii])
            # Alternatively, copy can be ensured via fancy indexing:
            # coeffs[[ii, imax]] = coeffs[[imax, ii]]
            # rhs[[ii, imax]] = rhs[[imax, ii]]
        if np.abs(coeffs[ii, ii]) < _DEPENDENCY_TOL:
            msg = "System of equations is linearly dependent"
            raise ValueError(msg)

        for jj in range(ii + 1, nn):
            coeff = -coeffs[jj, ii] / coeffs[ii, ii]
            coeffs[jj, ii:] += coeff * coeffs[ii, ii:]
            rhs[jj] += coeff * rhs[ii]

    xx = np.empty((nn,), dtype=float)
    for ii in range(nn - 1, -1, -1):
        # Note: dot product of two arrays of zero size is 0.0
        xx[ii] = (rhs[ii] - coeffs[ii, ii + 1 :] @ xx[ii + 1 :]) / coeffs[ii, ii]
    return xx

"""Routines for solving a linear system of equations."""
import numpy as np


def gaussian_eliminate(coeffs, rhs):
    """Solves a linear system of equations (Ax = b) by Gauss-elimination

    Args:
        coeffs: Matrix with the coefficients. Shape: (n, n).
        rhs: Right hand side of the equation. Shape: (n,)

    Returns:
        Vector xx with the solution of the linear equation or None if the equations are linearly
        dependent.
    """
    nn = coeffs.shape[0]
    for ii in range(nn - 1):
        for jj in range(ii + 1, nn):
            coeff = -coeffs[jj, ii] / coeffs[ii, ii]
            coeffs[jj, ii:] += coeff * coeffs[ii, ii:]
            rhs[jj] += coeff * rhs[ii]

    xx = np.empty((nn,), dtype=float)
    for ii in range(nn - 1, -1, -1):
        # Note: dot product of two arrays of zero size is 0.0
        xx[ii] = (rhs[ii] - coeffs[ii, ii + 1 :] @ xx[ii + 1 :]) / coeffs[ii, ii]
    return xx

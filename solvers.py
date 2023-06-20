"""Routines for solving a linear system of equations."""
import numpy as np

# Tolerance value for dependency check
_DEPENDENCY_TOL = 1e-10


def solve(coeffs, rhs):
    """Solves a linear system of equations (Ax = b) by Gauss-elimination

    Args:
        coeffs: Matrix with the coefficients. Shape: (n, n).
        rhs: Right hand side of the equation. Shape: (n,)

    Returns:
        Vector xx with the solution of the linear equation.

    Raises:
        numpy.linalg.LinAlgError: if the system of equation is linearly dependent.
    """
    return np.linalg.solve(coeffs, rhs)

"""I/O routines for the linsolver package"""
import numpy as np


def read_input(inputfile):
    """Reads an input file for the linear solver.

    Args:
        inputfile: Name of the file to read

    Returns:
        Tuple with an NxN matrix (the coefficient matrix) and an (N,) vector (right hand side).
    """
    coeffs_rhs = np.loadtxt(inputfile)
    nn = coeffs_rhs.shape[1]
    coeffs = np.array(coeffs_rhs[:nn, :])
    rhs = np.array(coeffs_rhs[nn, :])
    return coeffs, rhs


def write_result(outputfile, result):
    """Writes the result of the solver to a file.

    Args:
        outputfile: Name of the result file.
        result: Vector containing the solver result
    """
    np.savetxt(outputfile, result)


def write_lindep_error(outputfile):
    """Writes error message about linear dependency.

    Args:
        outputfile: Name of the result file.
    """
    with open(outputfile, "w", encoding="utf-8") as fp:
        fp.write("ERROR::LINDEP: Linearly dependent equations\n")

import numpy
import solvers

# Dimension of the system of equations to solve
_EQUATION_DIM = 2000


def main():
    """Main script"""
    coeffs = numpy.random.random((_EQUATION_DIM, _EQUATION_DIM))
    rhs = numpy.random.random((_EQUATION_DIM,))
    _ = solvers.solve(coeffs, rhs)


if __name__ == "__main__":
    main()

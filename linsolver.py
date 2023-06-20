"""Reads a linear system of equations and solves it and writes result."""
import sys
import linsolverio
import solvers

INPUT_FILE_NAME = "linsolver.in"

OUTPUT_FILE_NAME = "linsolver.out"


def main():
    """Main script."""
    try:
        coeffs, rhs = linsolverio.read_input(INPUT_FILE_NAME)
    except IOError:
        print(f"Could not read file '{INPUT_FILE_NAME}'")
        sys.exit(1)

    try:
        result = solvers.gaussian_eliminate(coeffs, rhs)
    except ValueError:
        result = None

    try:
        if result is None:
            linsolverio.write_lindep_error(OUTPUT_FILE_NAME)
        else:
            linsolverio.write_result(OUTPUT_FILE_NAME, result)
    except IOError:
        print(f"Could not write file '{OUTPUT_FILE_NAME}'")
        sys.exit(1)


if __name__ == "__main__":
    main()

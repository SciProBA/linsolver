"""Contains routines to test the solvers module"""

from pathlib import Path
import pytest
import numpy as np
import solvers


ABSOLUTE_TOLERANCE = 1e-10

RELATIVE_TOLERANCE = 1e-10

TEST_DATA_PATH = Path("testdata")

SUCCESSFUL_TESTS = ["simple", "needs_pivot"]

LINEAR_DEP_TESTS = ["linearly_dependant"]


@pytest.mark.parametrize("testname", SUCCESSFUL_TESTS)
def test_successful_elimination(testname):
    """Tests successful elimination"""
    aa, bb = _get_input(testname)
    xx_expected = _get_expected_output(testname)
    xx_gauss = solvers.gaussian_eliminate(aa, bb)
    assert np.allclose(xx_gauss, xx_expected, atol=ABSOLUTE_TOLERANCE, rtol=RELATIVE_TOLERANCE)


@pytest.mark.parametrize("testname", LINEAR_DEP_TESTS)
def test_linear_dependancy(testname):
    """Tests linear dependancy"""
    aa, bb = _get_input(testname)
    with pytest.raises(ValueError):
        _ = solvers.gaussian_eliminate(aa, bb)


def _get_input(testname):
    """Reads the input for a given test"""
    testinfile = TEST_DATA_PATH / (testname + ".in")
    data = np.loadtxt(testinfile)
    nn = data.shape[1]
    aa = data[:nn, :]
    bb = data[nn, :]
    return aa, bb


def _get_expected_output(testname):
    """Reads the ouput for a given test"""
    testoutfile = TEST_DATA_PATH / (testname + ".out")
    data = np.loadtxt(testoutfile)
    return data


# In case the python script is executed directly, it should start pytest.
if __name__ == "__main__":
    pytest.main()

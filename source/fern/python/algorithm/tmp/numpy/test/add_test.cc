#define BOOST_TEST_MODULE fern fern_python_extension_algorithm_numpy_add
#include <Python.h>
#include <boost/test/unit_test.hpp>
#include "fern/core/string.h"


int run_python_snippet(
    std::vector<fern::String> const& statements)
{
    std::string script{fern::join(statements, "\n").encode_in_utf8()};
    return PyRun_SimpleString(script.c_str());
}


class PythonInterpreterClient
{

public:

    PythonInterpreterClient()
    {
        Py_Initialize();

        int result{run_python_snippet({
            "import numpy as np",
            "import numpy.ma as ma",
            "from fern.algorithm.numpy import add",
        })};

        BOOST_REQUIRE_EQUAL(result, 0);
    }

    ~PythonInterpreterClient()
    {
        Py_Finalize();
    }

    bool test_statements(
        std::vector<fern::String> const& statements)
    {
        int result{run_python_snippet(statements)};
        // BOOST_CHECK_EQUAL(result, 0);
        return result == 0;
    }

};


BOOST_FIXTURE_TEST_SUITE(add, PythonInterpreterClient)

BOOST_AUTO_TEST_CASE(overloads)
{
    // Int - Float
    BOOST_CHECK(test_statements({
        "add(4, 5)",
        "add(4.5, 5.4)",
        "add(4, 5.4)",
        "add(4.5, 5)",
    }));

    // Int - Numpy array
    BOOST_CHECK(test_statements({
        "add(4, np.array([1, 2, 3]))",
        "add(np.array([1, 2, 3]), 4)",
    }));

    // Float - Numpy array
    BOOST_CHECK(test_statements({
        "add(4.5, np.array([1, 2, 3]))",
        "add(np.array([1, 2, 3]), 4.5)",
    }));

    // Int - Masked Numpy array
    BOOST_CHECK(test_statements({
        "add(4, ma.masked_array([1, 2, 3], [False, True, False]))",
        "add(ma.masked_array([1, 2, 3], [False, True, False]), 4)",
    }));

    // Float - Masked Numpy array
    BOOST_CHECK(test_statements({
        "add(4.5, ma.masked_array([1, 2, 3], [False, True, False]))",
        "add(ma.masked_array([1, 2, 3], [False, True, False]), 4.5)",
    }));

    // Numpy array - Masked Numpy array
    BOOST_CHECK(test_statements({
        "add(np.array([1, 2, 3]), "
            "ma.masked_array([1, 2, 3], [False, True, False]))",
        "add(ma.masked_array([1, 2, 3], [False, True, False]), "
            "np.array([1, 2, 3]))",
    }));
}

BOOST_AUTO_TEST_SUITE_END()

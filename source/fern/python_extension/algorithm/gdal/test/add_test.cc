#define BOOST_TEST_MODULE fern fern_python_extension_algorithm_gdal_add
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
            "from osgeo import gdal",
            "from osgeo.gdalconst import *",
            "from fern.algorithm.gdal import add",

            "int_ = 4",
            "float_ = 5.4",

            "array = np.array([1, 2, 3])",

            "driver = gdal.GetDriverByName(\"MEM\")",
            "nr_rows = 3",
            "nr_cols = 2",
            "value_type = GDT_Int32",
            "nr_bands = 1",
            "dataset = driver.Create(\"\", nr_cols, nr_rows, nr_bands, "
                "value_type)",
            "raster_band = dataset.GetRasterBand(1)",
            "array = np.random.randint(0, 255, (nr_rows, nr_cols)).astype("
                "np.int32)",
            "raster_band.WriteArray(array)",
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
        "add(int_, int_)",
        "add(float_, float_)",
        "add(int_, float_)",
        "add(float_, int_)",
    }));

    // Int - Numpy array
    BOOST_CHECK(test_statements({
        "add(int_, array)",
        "add(array, int_)",
    }));

    // Float - Numpy array
    BOOST_CHECK(test_statements({
        "add(float_, array)",
        "add(array, float_)",
    }));

    // Numpy array - Numpy array
    BOOST_CHECK(test_statements({
        "add(array, array)",
    }));

    // Int - GDAL raster band
    BOOST_CHECK(test_statements({
        "add(int_, raster_band)",
        "add(raster_band, int_)",
    }));

    // Float - GDAL raster band
    BOOST_CHECK(test_statements({
        "add(float_, raster_band)",
        "add(raster_band, float_)",
    }));

    // GDAL raster band - GDAL raster band
    BOOST_CHECK(test_statements({
        "add(raster_band, raster_band)",
    }));

    // TODO
    // Numpy array - GDAL raster band
    // BOOST_CHECK(test_statements({
    //     "add(array, raster_band)",
    //     "add(raster_band, array)",
    // }));
}

BOOST_AUTO_TEST_SUITE_END()

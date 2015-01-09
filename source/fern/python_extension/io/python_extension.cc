#include <boost/python.hpp>
#include <gdal_priv.h>
#include "fern/python_extension/core/init_python_module.h"
#include "fern/python_extension/io/gdal.h"


namespace bp = boost::python;
namespace fp = fern::python;

BOOST_PYTHON_MODULE(_fern_io)
{
    INIT_PYTHON_MODULE("C++ module with I/O related functionality.")

    // Don't throw in case of an error.
    CPLSetErrorHandler(CPLQuietErrorHandler);

    GDALAllRegister();

    bp::def("read_raster", fp::read_raster,
        "Read raster given the name passed in and return a reference to\n"
        "the result.",
        bp::arg("name"))
        ;
    bp::def("write_raster", fp::write_raster,
        "Write raster given the raster, name and format passed in.",
        (bp::arg("raster"), bp::arg("name"), bp::arg("format")))
        ;
}

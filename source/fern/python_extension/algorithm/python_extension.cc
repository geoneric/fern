#include <boost/python.hpp>
#include "fern/core/thread_client.h"
#include "fern/python_extension/core/init_python_module.h"
#include "fern/python_extension/algorithm/add.h"
#include "fern/python_extension/algorithm/slope.h"


namespace bp = boost::python;
namespace fp = fern::python;


static fern::ThreadClient client;


fp::MaskedRasterHandle (*add_raster_raster)(
    fp::MaskedRasterHandle const&,
    fp::MaskedRasterHandle const&) = &fp::add;
fp::MaskedRasterHandle (*add_int64_raster)(
    int64_t,
    fp::MaskedRasterHandle const&) = &fp::add;
fp::MaskedRasterHandle (*add_raster_int64)(
    fp::MaskedRasterHandle const&,
    int64_t) = &fp::add;
fp::MaskedRasterHandle (*add_double_raster)(
    double,
    fp::MaskedRasterHandle const&) = &fp::add;
fp::MaskedRasterHandle (*add_raster_double)(
    fp::MaskedRasterHandle const&,
    double) = &fp::add;


BOOST_PYTHON_MODULE(_fern_algorithm)
{
    INIT_PYTHON_MODULE("C++ module with algorithms.")

    bp::def("slope", fp::slope,
        "Calculate the slope and return the result.",
        bp::arg("dem"))
        ;

    bp::def("iadd", fp::iadd,
        bp::return_value_policy<bp::reference_existing_object>(),
        "Add the second operand to the first and return the result.")
        ;

    bp::def("add", add_raster_raster,
        "Add the rasters and return the result.")
        ;
    bp::def("add", add_raster_double,
        "Add the raster and the number and return the result.")
        ;
    bp::def("add", add_double_raster,
        "Add the raster and the number and return the result.")
        ;
    bp::def("add", add_raster_int64,
        "Add the raster and the number and return the result.")
        ;
    bp::def("add", add_int64_raster,
        "Add the raster and the number and return the result.")
        ;
}

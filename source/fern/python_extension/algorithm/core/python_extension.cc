#include <boost/python.hpp>
#include "fern/python_extension/core/init_python_module.h"
#include "fern/python_extension/algorithm/core/if.h"


namespace bp = boost::python;
namespace fp = fern::python;


fp::MaskedRasterHandle (*if_raster_raster)(
    fp::MaskedRasterHandle const&,
    fp::MaskedRasterHandle const&,
    fp::MaskedRasterHandle const&) = &fp::if_;
fp::MaskedRasterHandle (*if_int64_raster)(
    fp::MaskedRasterHandle const&,
    int64_t,
    fp::MaskedRasterHandle const&) = &fp::if_;
fp::MaskedRasterHandle (*if_raster_int64)(
    fp::MaskedRasterHandle const&,
    fp::MaskedRasterHandle const&,
    int64_t) = &fp::if_;
fp::MaskedRasterHandle (*if_double_raster)(
    fp::MaskedRasterHandle const&,
    double,
    fp::MaskedRasterHandle const&) = &fp::if_;
fp::MaskedRasterHandle (*if_raster_double)(
    fp::MaskedRasterHandle const&,
    fp::MaskedRasterHandle const&,
    double) = &fp::if_;


BOOST_PYTHON_MODULE(_fern_algorithm_core)
{
    INIT_PYTHON_MODULE("C++ module with core algorithms.")

    // If
    bp::def("if_", if_raster_raster,
        "Conditionally assign cells from true or false raster to\n"
        "the result.")
        ;
    bp::def("if_", if_raster_double,
        "Conditionally assign cell from true raster or false value to\n"
        "the result.")
        ;
    bp::def("if_", if_double_raster,
        "Conditionally assign cell from true value or false raster to\n"
        "the result.")
        ;
    bp::def("if_", if_raster_int64,
        "Conditionally assign cell from true raster or false value to\n"
        "the result.")
        ;
    bp::def("if_", if_int64_raster,
        "Conditionally assign cell from true value or false raster to\n"
        "the result.")
        ;
}

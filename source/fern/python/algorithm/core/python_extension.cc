#include <boost/python.hpp>
#include "fern/python/core/init_python_module.h"
#include "fern/python/algorithm/core/if.h"


namespace bp = boost::python;
namespace fa = fern::algorithm;
namespace fp = fern::python;


fp::MaskedRasterHandle (*if_raster_raster)(
    fa::ExecutionPolicy&,
    fp::MaskedRasterHandle const&,
    fp::MaskedRasterHandle const&,
    fp::MaskedRasterHandle const&) = &fp::if_;
fp::MaskedRasterHandle (*if_int64_raster)(
    fa::ExecutionPolicy&,
    fp::MaskedRasterHandle const&,
    int64_t,
    fp::MaskedRasterHandle const&) = &fp::if_;
fp::MaskedRasterHandle (*if_raster_int64)(
    fa::ExecutionPolicy&,
    fp::MaskedRasterHandle const&,
    fp::MaskedRasterHandle const&,
    int64_t) = &fp::if_;
fp::MaskedRasterHandle (*if_double_raster)(
    fa::ExecutionPolicy&,
    fp::MaskedRasterHandle const&,
    double,
    fp::MaskedRasterHandle const&) = &fp::if_;
fp::MaskedRasterHandle (*if_raster_double)(
    fa::ExecutionPolicy&,
    fp::MaskedRasterHandle const&,
    fp::MaskedRasterHandle const&,
    double) = &fp::if_;


BOOST_PYTHON_MODULE(_fern_algorithm_core)
{
    INIT_PYTHON_MODULE("_fern_algorithm_core")

    bp::def("if_", if_raster_raster);
    bp::def("if_", if_raster_int64);
    bp::def("if_", if_int64_raster);
    bp::def("if_", if_raster_double);
    bp::def("if_", if_double_raster);
}

// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#include <boost/python.hpp>
#include "fern/python/core/init_python_module.h"
#include "fern/python/algorithm/algebra/elementary/add.h"
#include "fern/python/algorithm/algebra/elementary/greater.h"
#include "fern/python/algorithm/algebra/elementary/less.h"
#include "fern/python/algorithm/algebra/elementary/multiply.h"


namespace bp = boost::python;
namespace fa = fern::algorithm;
namespace fp = fern::python;


fp::MaskedRasterHandle (*add_raster_raster)(
    fa::ExecutionPolicy&,
    fp::MaskedRasterHandle const&,
    fp::MaskedRasterHandle const&) = &fp::add;
fp::MaskedRasterHandle (*add_int64_raster)(
    fa::ExecutionPolicy&,
    int64_t,
    fp::MaskedRasterHandle const&) = &fp::add;
fp::MaskedRasterHandle (*add_raster_int64)(
    fa::ExecutionPolicy&,
    fp::MaskedRasterHandle const&,
    int64_t) = &fp::add;
fp::MaskedRasterHandle (*add_double_raster)(
    fa::ExecutionPolicy&,
    double,
    fp::MaskedRasterHandle const&) = &fp::add;
fp::MaskedRasterHandle (*add_raster_double)(
    fa::ExecutionPolicy&,
    fp::MaskedRasterHandle const&,
    double) = &fp::add;


fp::MaskedRasterHandle (*multiply_raster_raster)(
    fa::ExecutionPolicy&,
    fp::MaskedRasterHandle const&,
    fp::MaskedRasterHandle const&) = &fp::multiply;
fp::MaskedRasterHandle (*multiply_int64_raster)(
    fa::ExecutionPolicy&,
    int64_t,
    fp::MaskedRasterHandle const&) = &fp::multiply;
fp::MaskedRasterHandle (*multiply_raster_int64)(
    fa::ExecutionPolicy&,
    fp::MaskedRasterHandle const&,
    int64_t) = &fp::multiply;
fp::MaskedRasterHandle (*multiply_double_raster)(
    fa::ExecutionPolicy&,
    double,
    fp::MaskedRasterHandle const&) = &fp::multiply;
fp::MaskedRasterHandle (*multiply_raster_double)(
    fa::ExecutionPolicy&,
    fp::MaskedRasterHandle const&,
    double) = &fp::multiply;


fp::MaskedRasterHandle (*less_raster_raster)(
    fa::ExecutionPolicy&,
    fp::MaskedRasterHandle const&,
    fp::MaskedRasterHandle const&) = &fp::less;
fp::MaskedRasterHandle (*less_int64_raster)(
    fa::ExecutionPolicy&,
    int64_t,
    fp::MaskedRasterHandle const&) = &fp::less;
fp::MaskedRasterHandle (*less_raster_int64)(
    fa::ExecutionPolicy&,
    fp::MaskedRasterHandle const&,
    int64_t) = &fp::less;
fp::MaskedRasterHandle (*less_double_raster)(
    fa::ExecutionPolicy&,
    double,
    fp::MaskedRasterHandle const&) = &fp::less;
fp::MaskedRasterHandle (*less_raster_double)(
    fa::ExecutionPolicy&,
    fp::MaskedRasterHandle const&,
    double) = &fp::less;


fp::MaskedRasterHandle (*greater_raster_raster)(
    fa::ExecutionPolicy&,
    fp::MaskedRasterHandle const&,
    fp::MaskedRasterHandle const&) = &fp::greater;
fp::MaskedRasterHandle (*greater_int64_raster)(
    fa::ExecutionPolicy&,
    int64_t,
    fp::MaskedRasterHandle const&) = &fp::greater;
fp::MaskedRasterHandle (*greater_raster_int64)(
    fa::ExecutionPolicy&,
    fp::MaskedRasterHandle const&,
    int64_t) = &fp::greater;
fp::MaskedRasterHandle (*greater_double_raster)(
    fa::ExecutionPolicy&,
    double,
    fp::MaskedRasterHandle const&) = &fp::greater;
fp::MaskedRasterHandle (*greater_raster_double)(
    fa::ExecutionPolicy&,
    fp::MaskedRasterHandle const&,
    double) = &fp::greater;


BOOST_PYTHON_MODULE(_fern_algorithm_algebra_elementary)
{
    INIT_PYTHON_MODULE("_fern_algorithm_algebra_elementary")

    bp::def("iadd", fp::iadd,
        bp::return_value_policy<bp::reference_existing_object>());
    bp::def("add", add_raster_raster);
    bp::def("add", add_raster_double);
    bp::def("add", add_double_raster);
    bp::def("add", add_raster_int64);
    bp::def("add", add_int64_raster);

    bp::def("imultiply", fp::imultiply,
        bp::return_value_policy<bp::reference_existing_object>());
    bp::def("multiply", multiply_raster_raster);
    bp::def("multiply", multiply_raster_double);
    bp::def("multiply", multiply_double_raster);
    bp::def("multiply", multiply_raster_int64);
    bp::def("multiply", multiply_int64_raster);

    bp::def("less", less_raster_raster);
    bp::def("less", less_raster_double);
    bp::def("less", less_double_raster);
    bp::def("less", less_raster_int64);
    bp::def("less", less_int64_raster);

    bp::def("greater", greater_raster_raster);
    bp::def("greater", greater_raster_double);
    bp::def("greater", greater_double_raster);
    bp::def("greater", greater_raster_int64);
    bp::def("greater", greater_int64_raster);
}

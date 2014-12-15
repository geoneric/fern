#include <boost/python.hpp>
#include "fern/python_extension/core/init_python_module.h"
#include "fern/python_extension/algorithm/algebra/elementary/add.h"
#include "fern/python_extension/algorithm/algebra/elementary/greater.h"
#include "fern/python_extension/algorithm/algebra/elementary/less.h"
#include "fern/python_extension/algorithm/algebra/elementary/multiply.h"


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
    INIT_PYTHON_MODULE("C++ module with elementary algebraic algorithms.")

    // Add.
    bp::def("iadd", fp::iadd,
        bp::return_value_policy<bp::reference_existing_object>(),
        "Add the raster to self inplace and return the result.")
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

    // Multiply.
    bp::def("imultiply", fp::imultiply,
        bp::return_value_policy<bp::reference_existing_object>(),
        "Multiply the raster to self inplace and return the result.")
        ;
    bp::def("multiply", multiply_raster_raster,
        "Multiply the rasters and return the result.")
        ;
    bp::def("multiply", multiply_raster_double,
        "Multiply the raster and the number and return the result.")
        ;
    bp::def("multiply", multiply_double_raster,
        "Multiply the raster and the number and return the result.")
        ;
    bp::def("multiply", multiply_raster_int64,
        "Multiply the raster and the number and return the result.")
        ;
    bp::def("multiply", multiply_int64_raster,
        "Multiply the raster and the number and return the result.")
        ;

    // Less
    bp::def("less", less_raster_raster,
        "Determine whether the first raster is less than the second\n"
        "and return the result.")
        ;
    bp::def("less", less_raster_double,
        "Determine whether the raster is less than the number\n"
        "and return the result.")
        ;
    bp::def("less", less_double_raster,
        "Determine whether the number is less than the raster\n"
        "and return the result.")
        ;
    bp::def("less", less_raster_int64,
        "Determine whether the raster is less than the number\n"
        "and return the result.")
        ;
    bp::def("less", less_int64_raster,
        "Determine whether the number is less than the raster\n"
        "and return the result.")
        ;

    // Greater
    bp::def("greater", greater_raster_raster,
        "Determine whether the first raster is greater than the second\n"
        "and return the result.")
        ;
    bp::def("greater", greater_raster_double,
        "Determine whether the raster is greater than the number\n"
        "and return the result.")
        ;
    bp::def("greater", greater_double_raster,
        "Determine whether the number is greater than the raster\n"
        "and return the result.")
        ;
    bp::def("greater", greater_raster_int64,
        "Determine whether the raster is greater than the number\n"
        "and return the result.")
        ;
    bp::def("greater", greater_int64_raster,
        "Determine whether the number is greater than the raster\n"
        "and return the result.")
        ;
}

#include <boost/python.hpp>
#include "fern/core/thread_client.h"
#include "fern/python_extension/core/init_python_module.h"
#include "fern/python_extension/algorithm/add.h"
#include "fern/python_extension/algorithm/greater.h"
#include "fern/python_extension/algorithm/if.h"
#include "fern/python_extension/algorithm/less.h"
#include "fern/python_extension/algorithm/multiply.h"
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


fp::MaskedRasterHandle (*multiply_raster_raster)(
    fp::MaskedRasterHandle const&,
    fp::MaskedRasterHandle const&) = &fp::multiply;
fp::MaskedRasterHandle (*multiply_int64_raster)(
    int64_t,
    fp::MaskedRasterHandle const&) = &fp::multiply;
fp::MaskedRasterHandle (*multiply_raster_int64)(
    fp::MaskedRasterHandle const&,
    int64_t) = &fp::multiply;
fp::MaskedRasterHandle (*multiply_double_raster)(
    double,
    fp::MaskedRasterHandle const&) = &fp::multiply;
fp::MaskedRasterHandle (*multiply_raster_double)(
    fp::MaskedRasterHandle const&,
    double) = &fp::multiply;


fp::MaskedRasterHandle (*less_raster_raster)(
    fp::MaskedRasterHandle const&,
    fp::MaskedRasterHandle const&) = &fp::less;
fp::MaskedRasterHandle (*less_int64_raster)(
    int64_t,
    fp::MaskedRasterHandle const&) = &fp::less;
fp::MaskedRasterHandle (*less_raster_int64)(
    fp::MaskedRasterHandle const&,
    int64_t) = &fp::less;
fp::MaskedRasterHandle (*less_double_raster)(
    double,
    fp::MaskedRasterHandle const&) = &fp::less;
fp::MaskedRasterHandle (*less_raster_double)(
    fp::MaskedRasterHandle const&,
    double) = &fp::less;


fp::MaskedRasterHandle (*greater_raster_raster)(
    fp::MaskedRasterHandle const&,
    fp::MaskedRasterHandle const&) = &fp::greater;
fp::MaskedRasterHandle (*greater_int64_raster)(
    int64_t,
    fp::MaskedRasterHandle const&) = &fp::greater;
fp::MaskedRasterHandle (*greater_raster_int64)(
    fp::MaskedRasterHandle const&,
    int64_t) = &fp::greater;
fp::MaskedRasterHandle (*greater_double_raster)(
    double,
    fp::MaskedRasterHandle const&) = &fp::greater;
fp::MaskedRasterHandle (*greater_raster_double)(
    fp::MaskedRasterHandle const&,
    double) = &fp::greater;


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


BOOST_PYTHON_MODULE(_fern_algorithm)
{
    INIT_PYTHON_MODULE("C++ module with algorithms.")

    bp::def("slope", fp::slope,
        "Calculate the slope and return the result.",
        bp::arg("dem"))
        ;

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

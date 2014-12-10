#include <boost/python.hpp>
#include "fern/core/thread_client.h"
#include "fern/python_extension/core/init_python_module.h"
#include "fern/python_extension/algorithm/add.h"
#include "fern/python_extension/algorithm/slope.h"


namespace bp = boost::python;
namespace fp = fern::python;


static fern::ThreadClient client;


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
    bp::def("add", fp::add,
        "Add the operands and return the result.")
        ;
}

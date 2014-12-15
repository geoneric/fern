#include <boost/python.hpp>
#include "fern/python_extension/core/init_python_module.h"
#include "fern/python_extension/algorithm/space/focal/slope.h"


namespace bp = boost::python;
namespace fp = fern::python;


BOOST_PYTHON_MODULE(_fern_algorithm_space_focal)
{
    INIT_PYTHON_MODULE("C++ module with space/focal algorithms.")

    bp::def("slope", fp::slope,
        "Calculate the slope and return the result.")
        ;
}

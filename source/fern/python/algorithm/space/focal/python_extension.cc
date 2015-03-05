#include <boost/python.hpp>
#include "fern/python/core/init_python_module.h"
#include "fern/python/algorithm/space/focal/slope.h"


namespace bp = boost::python;
namespace fp = fern::python;


BOOST_PYTHON_MODULE(_fern_algorithm_space_focal)
{
    INIT_PYTHON_MODULE("_fern_algorithm_space_focal")

    bp::def("slope", fp::slope);
}

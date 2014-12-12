#include <boost/python.hpp>
// #include "fern/core/thread_client.h"
#include "fern/algorithm/policy/execution_policy.h"
#include "fern/python_extension/core/init_python_module.h"


namespace bp = boost::python;


// static fern::ThreadClient thread_client;
static fern::algorithm::ExecutionPolicy execution_policy =
    fern::algorithm::sequential;


BOOST_PYTHON_MODULE(_fern_algorithm)
{
    INIT_PYTHON_MODULE("C++ module with algorithms.")
}

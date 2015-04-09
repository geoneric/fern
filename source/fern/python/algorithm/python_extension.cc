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
#include "fern/python/algorithm/execution_policy.h"


namespace bp = boost::python;
namespace fa = fern::algorithm;
namespace fp = fern::python;


BOOST_PYTHON_MODULE(_fern_algorithm)
{
    INIT_PYTHON_MODULE("_fern_algorithm")

    // bp::class_<fern::ExecutionPolicy, boost::noncopyable>(
    //     "ExecutionPolicy",
    //     "Execution policy used by algorithms.\n"
    //     "\n"
    //     "An execution policy determines whether or not algorithms\n"
    //     "execute on a single CPU core (sequential) or on multiple\n"
    //     "CPU cores (parallel).",


    bp::class_<fa::SequentialExecutionPolicy>(
        "SequentialExecutionPolicy",
        "Sequential execution policy");
    bp::class_<fa::ParallelExecutionPolicy>(
        "ParallelExecutionPolicy",
        "Parallel execution policy");
    bp::class_<fa::ExecutionPolicy>(
        "ExecutionPolicy",
        "Execution policy");

    bp::def("set_execution_policy", fp::set_execution_policy,
        "Set the execution policy passed in as the one algorithms\n"
        "should use from now on.");
    bp::def("execution_policy", fp::execution_policy,
        bp::return_value_policy<bp::reference_existing_object>(),
        "Return the currently set execution policy.");

    bp::implicitly_convertible<fa::SequentialExecutionPolicy,
       fa::ExecutionPolicy>();
    bp::implicitly_convertible<fa::ParallelExecutionPolicy,
       fa::ExecutionPolicy>();
}

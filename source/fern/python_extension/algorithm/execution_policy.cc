#include "fern/python_extension/algorithm/execution_policy.h"


namespace fern {
namespace python {

static algorithm::ExecutionPolicy _execution_policy{algorithm::sequential};


void set_execution_policy(
    algorithm::ExecutionPolicy const& execution_policy)
{
    _execution_policy = execution_policy;
}


algorithm::ExecutionPolicy const& execution_policy()
{
    return _execution_policy;
}

} // namespace python
} // namespace fern

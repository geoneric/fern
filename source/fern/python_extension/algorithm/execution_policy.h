#pragma once
#include "fern/algorithm/policy/execution_policy.h"


namespace fern {
namespace python {

void               set_execution_policy(algorithm::ExecutionPolicy const&
                                            execution_policy);

algorithm::ExecutionPolicy const&
                   execution_policy    ();

} // namespace python
} // namespace fern

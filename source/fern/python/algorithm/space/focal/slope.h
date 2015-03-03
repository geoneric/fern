#pragma once
#include "fern/algorithm/policy/execution_policy.h"
#include "fern/python/feature/masked_raster.h"


namespace fern {
namespace python {

MaskedRasterHandle slope               (algorithm::ExecutionPolicy&
                                            execution_policy,
                                        MaskedRasterHandle const& dem);

} // namespace python
} // namespace fern

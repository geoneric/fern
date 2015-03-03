#pragma once
#include "fern/algorithm/policy/execution_policy.h"
#include "fern/python/feature/masked_raster.h"


namespace fern {
namespace python {

MaskedRasterHandle if_                 (algorithm::ExecutionPolicy&
                                            execution_policy,
                                        MaskedRasterHandle const& condition,
                                        MaskedRasterHandle const& true_value,
                                        MaskedRasterHandle const& false_value);

MaskedRasterHandle if_                 (algorithm::ExecutionPolicy&
                                            execution_policy,
                                        MaskedRasterHandle const& condition,
                                        int64_t true_value,
                                        MaskedRasterHandle const& false_value);

MaskedRasterHandle if_                 (algorithm::ExecutionPolicy&
                                            execution_policy,
                                        MaskedRasterHandle const& condition,
                                        MaskedRasterHandle const& true_value,
                                        int64_t false_value);

MaskedRasterHandle if_                 (algorithm::ExecutionPolicy&
                                            execution_policy,
                                        MaskedRasterHandle const& condition,
                                        double true_value,
                                        MaskedRasterHandle const& false_value);

MaskedRasterHandle if_                 (algorithm::ExecutionPolicy&
                                            execution_policy,
                                        MaskedRasterHandle const& condition,
                                        MaskedRasterHandle const& true_value,
                                        double false_value);

} // namespace python
} // namespace fern

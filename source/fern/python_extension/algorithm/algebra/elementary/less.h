#pragma once
#include "fern/algorithm/policy/execution_policy.h"
#include "fern/python_extension/feature/masked_raster.h"


namespace fern {
namespace python {

MaskedRasterHandle less                (algorithm::ExecutionPolicy&
                                            execution_policy,
                                        MaskedRasterHandle const& raster1,
                                        MaskedRasterHandle const& raster2);

MaskedRasterHandle less                (algorithm::ExecutionPolicy&
                                            execution_policy,
                                        int64_t value,
                                        MaskedRasterHandle const& raster);

MaskedRasterHandle less                (algorithm::ExecutionPolicy&
                                            execution_policy,
                                        MaskedRasterHandle const& raster,
                                        int64_t value);

MaskedRasterHandle less                (algorithm::ExecutionPolicy&
                                            execution_policy,
                                        double value,
                                        MaskedRasterHandle const& raster);

MaskedRasterHandle less                (algorithm::ExecutionPolicy&
                                            execution_policy,
                                        MaskedRasterHandle const& raster,
                                        double value);

} // namespace python
} // namespace fern

#pragma once
#include "fern/algorithm/policy/execution_policy.h"
#include "fern/python/feature/masked_raster.h"


namespace fern {
namespace python {

MaskedRasterHandle&
                   iadd                (algorithm::ExecutionPolicy&
                                            execution_policy,
                                        MaskedRasterHandle& self,
                                        MaskedRasterHandle const& other);

MaskedRasterHandle add                 (algorithm::ExecutionPolicy&
                                            execution_policy,
                                        MaskedRasterHandle const& raster1,
                                        MaskedRasterHandle const& raster2);

MaskedRasterHandle add                 (algorithm::ExecutionPolicy&
                                            execution_policy,
                                        int64_t value,
                                        MaskedRasterHandle const& raster);

MaskedRasterHandle add                 (algorithm::ExecutionPolicy&
                                            execution_policy,
                                        MaskedRasterHandle const& raster,
                                        int64_t value);

MaskedRasterHandle add                 (algorithm::ExecutionPolicy&
                                            execution_policy,
                                        double value,
                                        MaskedRasterHandle const& raster);

MaskedRasterHandle add                 (algorithm::ExecutionPolicy&
                                            execution_policy,
                                        MaskedRasterHandle const& raster,
                                        double value);

} // namespace python
} // namespace fern

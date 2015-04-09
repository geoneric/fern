// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
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

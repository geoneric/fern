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

MaskedRasterHandle slope               (algorithm::ExecutionPolicy&
                                            execution_policy,
                                        MaskedRasterHandle const& dem);

} // namespace python
} // namespace fern

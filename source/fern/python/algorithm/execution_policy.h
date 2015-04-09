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


namespace fern {
namespace python {

void               set_execution_policy(algorithm::ExecutionPolicy const&
                                            execution_policy);

algorithm::ExecutionPolicy const&
                   execution_policy    ();

} // namespace python
} // namespace fern

// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include "fern/algorithm/core/operation_categories.h"


namespace fern {
namespace algorithm {

template<
    typename Operation>
struct OperationTraits
{
    using category = typename Operation::category;
};

} // namespace algorithm
} // namespace fern

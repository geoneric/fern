// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include "fern/feature/core/array.h"
#include "fern/algorithm/core/mask_customization_point.h"


namespace fern {
namespace algorithm {

template<
    size_t nr_dimensions>
inline constexpr bool no_data_value(
    Array<bool, nr_dimensions> const& /* mask */)
{
    return true;
}

} // namespace algorithm
} // namespace fern

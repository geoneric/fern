// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include <cstddef>
#include "fern/core/data_type_traits.h"
#include "fern/algorithm/core/index_ranges.h"


namespace fern {

template<
    size_t nr_dimensions>
struct DataTypeTraits<
    algorithm::IndexRanges<nr_dimensions>>
{

    /// using value_type = Coordinate;

    /// using reference = value_type&;

    /// using const_reference = value_type const&;

    static size_t const rank = nr_dimensions;

};


template<
    size_t nr_dimensions>
inline size_t size(
    algorithm::IndexRanges<nr_dimensions> const& ranges)
{
    return ranges.size();
}

} // namespace fern

#pragma once
#include <cstddef>
#include "fern/core/data_traits.h"
#include "fern/algorithm/core/index_ranges.h"


namespace fern {

template<
    size_t nr_dimensions>
struct DataTraits<
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

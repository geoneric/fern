#pragma once
#include "fern/feature/core/array.h"
#include "fern/algorithm/core/mask_customization_point.h"


namespace fern {
namespace algorithm {

// template<
//     typename T,
//     size_t nr_dimensions>
// T                  no_data_value       (Array<T, nr_dimensions> const& array);


template<
    size_t nr_dimensions>
inline constexpr bool no_data_value(
    Array<bool, nr_dimensions> const& /* mask */)
{
    return true;
}

} // namespace algorithm
} // namespace fern

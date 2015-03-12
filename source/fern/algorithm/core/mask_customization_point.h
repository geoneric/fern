#pragma once
#include "fern/core/data_traits.h"


namespace fern {
namespace algorithm {

template<
    typename Mask>
value_type<Mask>   no_data_value       (Mask const& mask);

} // namespace algorithm
} // namespace fern

#pragma once
#include "fern/core/data_traits.h"


namespace fern {
namespace algorithm {

/*!
    @ingroup    fern_algorithm_mask_customization_point_group
    @brief      Return the value used to signal a masked elements in @a mask.
*/
template<
    typename Mask>
value_type<Mask>   no_data_value       (Mask const& mask);

} // namespace algorithm
} // namespace fern

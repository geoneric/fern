// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include "fern/core/data_type_traits.h"


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

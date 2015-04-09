// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include "fern/algorithm/core/argument_traits.h"


namespace fern {
namespace algorithm {

/*!
    @ingroup    fern_algorithm_argument_customization_point_group
    @brief      Return a reference to the mask of \a argument.
    @sa         MaskT<Argument>
*/
template<
    typename Argument>
MaskT<Argument>&   mask                (Argument& argument);

} // namespace algorithm
} // namespace fern

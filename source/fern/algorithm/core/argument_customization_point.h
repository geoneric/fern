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

#pragma once
#include "fern/algorithm/core/argument_traits.h"


namespace fern {
namespace algorithm {

template<
    typename Argument>
MaskT<Argument>&   mask                (Argument& argument);

} // namespace algorithm
} // namespace fern

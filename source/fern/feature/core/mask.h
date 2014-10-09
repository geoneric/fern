#pragma once
#include "fern/feature/core/array.h"


namespace fern {

template<
    size_t nr_dimensions>
using Mask = Array<bool, nr_dimensions>;

} // namespace fern

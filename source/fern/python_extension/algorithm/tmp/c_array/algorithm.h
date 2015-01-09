#pragma once
#include "fern/core/clone.h"
#include "fern/feature/core/array_reference_traits.h"
#include "fern/algorithm.h"


namespace fern {

template<
    class U,
    class V,
    class W,
    size_t nr_dimensions>
inline void add(
    ArrayReference<W, nr_dimensions>& result,
    ArrayReference<U, nr_dimensions> const& array,
    V const& value)
{
    algorithm::algebra::add(algorithm::parallel, array, value, result);
}

} // namespace fern

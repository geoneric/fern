#pragma once
#include "fern/algorithm/core/argument_customization_point.h"
#include "fern/python/feature/detail/argument_traits/masked_raster.h"


namespace fern {
namespace algorithm {

template<
    typename T>
inline MaskT<python::detail::MaskedRaster<T>>& mask(
    python::detail::MaskedRaster<T>& argument)
{
    return argument;
}

} // namespace algorithm
} // namespace fern

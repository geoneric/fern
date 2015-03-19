#pragma once
#include "fern/feature/core/masked_raster.h"
#include "fern/algorithm/core/argument_customization_point.h"


namespace fern {
namespace algorithm {

template<
    typename T,
    size_t nr_dimensions>
inline MaskT<MaskedRaster<T, nr_dimensions>>& mask(
    MaskedRaster<T, nr_dimensions>& argument)
{
    return argument.mask();
}

} // namespace algorithm
} // namespace fern

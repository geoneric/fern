#pragma once
#include "fern/algorithm/core/argument_customization_point.h"
#include "fern/example/algorithm/raster.h"


namespace fern {
namespace algorithm {

template<
    typename T>
inline MaskT<example::Raster<T>>& mask(
    example::Raster<T>& argument)
{
    return argument;
}

} // namespace algorithm
} // namespace fern

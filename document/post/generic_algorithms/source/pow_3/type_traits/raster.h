#pragma once
#include "type_traits.h"


template<
    typename T,
    size_t nr_dimensions>
struct TypeTraits<
    fern::Raster<T, nr_dimensions>>
{
    using value_type = T;
};

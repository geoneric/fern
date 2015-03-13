#pragma once
#include <cassert>
#include "fern/core/data_customization_point.h"
#include "fern/core/data_traits/array_reference.h"


namespace fern {

template<
    class T,
    size_t nr_dimensions>
inline size_t size(
    ArrayReference<T, nr_dimensions> const& array)
{
    return array.num_elements();
}


template<
    class T,
    size_t nr_dimensions>
inline size_t size(
    ArrayReference<T, nr_dimensions> const& array,
    size_t dimension)
{
    assert(dimension < array.num_dimensions());
    return array.shape()[dimension];
}


template<
    class T,
    size_t nr_dimensions>
inline typename DataTraits<ArrayReference<T, nr_dimensions>>
        ::const_reference get(
    ArrayReference<T, nr_dimensions> const& array,
    size_t index)
{
    assert(index < array.num_elements());
    return array.data()[index];
}


template<
    class T,
    size_t nr_dimensions>
inline typename DataTraits<ArrayReference<T, nr_dimensions>>::reference get(
    ArrayReference<T, nr_dimensions>& array,
    size_t index)
{
    assert(index < array.num_elements());
    return array.data()[index];
}

} // namespace fern

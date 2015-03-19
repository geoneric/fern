#pragma once
#include <cassert>
#include "fern/core/data_customization_point.h"
#include "fern/feature/core/data_traits/array_view.h"


namespace fern {

template<
    typename T,
    size_t nr_dimensions>
inline size_t size(
    ArrayView<T, nr_dimensions> const& view)
{
    return view.num_elements();
}


template<
    typename T,
    size_t nr_dimensions>
inline size_t size(
    ArrayView<T, nr_dimensions> const& view,
    size_t dimension)
{
    assert(dimension < view.num_dimensions());
    return view.shape()[dimension];
}


template<
    typename T,
    size_t nr_dimensions>
inline typename DataTraits<ArrayView<T, nr_dimensions>>::const_reference get(
    ArrayView<T, nr_dimensions> const& view,
    size_t index)
{
    assert(index < view.num_elements());
    return view.data()[index];
}


template<
    typename T,
    size_t nr_dimensions>
inline typename DataTraits<ArrayView<T, nr_dimensions>>::reference get(
    ArrayView<T, nr_dimensions>& view,
    size_t index)
{
    assert(index < view.num_elements());
    return view.data()[index];
}

} // namespace fern

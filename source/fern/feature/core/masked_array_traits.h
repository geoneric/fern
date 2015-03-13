#pragma once
#include "fern/feature/core/data_customization_point/array.h"
#include "fern/feature/core/masked_array.h"


namespace fern {

template<
    typename T,
    size_t nr_dimensions>
struct DataTraits<
    MaskedArray<T, nr_dimensions>>
{

    using argument_category = typename detail::dispatch::ArrayCategoryTag<T,
        nr_dimensions>::type;

    template<
        typename U>
    struct Collection
    {
        using type = MaskedArray<U, nr_dimensions>;
    };

    template<
        typename U>
    struct Clone
    {
        using type = MaskedArray<U, nr_dimensions>;
    };

    using value_type = T;

    using reference = T&;

    using const_reference = T const&;

    static bool const is_masking = true;

    static size_t const rank = nr_dimensions;

};


template<
    typename T,
    size_t nr_dimensions>
inline size_t size(
    MaskedArray<T, nr_dimensions> const& array)
{
    return array.num_elements();
}


template<
    typename T,
    size_t nr_dimensions>
inline size_t size(
    MaskedArray<T, nr_dimensions> const& array,
    size_t dimension)
{
    assert(dimension < array.num_dimensions());
    return array.shape()[dimension];
}


template<
    typename T,
    size_t nr_dimensions>
inline size_t index(
    MaskedArray<T, nr_dimensions> const& array,
    size_t index1,
    size_t index2)
{
    return index1 * size(array, 1) + index2;
}


template<
    typename T,
    size_t nr_dimensions>
inline typename DataTraits<MaskedArray<T, nr_dimensions>>::const_reference
        get(
    MaskedArray<T, nr_dimensions> const& array,
    size_t index)
{
    assert(index < array.num_elements());
    return array.data()[index];
}


template<
    typename T,
    size_t nr_dimensions>
inline typename DataTraits<MaskedArray<T, nr_dimensions>>::reference get(
    MaskedArray<T, nr_dimensions>& array,
    size_t index)
{
    assert(index < array.num_elements());
    return array.data()[index];
}


template<
    typename U,
    typename V>
inline MaskedArray<U, 1> clone(
    MaskedArray<V, 1> const& array)
{
    return std::move(MaskedArray<U, 1>(extents[array.shape()[0]]));
}


template<
    typename U,
    typename V>
inline MaskedArray<U, 1> clone(
    MaskedArray<V, 1> const& array,
    U const& value)
{
    return std::move(MaskedArray<U, 1>(extents[array.shape()[0]], value));
}


template<
    typename U,
    typename V>
inline MaskedArray<U, 2> clone(
    MaskedArray<V, 2> const& array)
{
    return std::move(MaskedArray<U, 2>(
        extents[array.shape()[0]][array.shape()[1]]));
}


template<
    typename U,
    typename V>
inline MaskedArray<U, 2> clone(
    MaskedArray<V, 2> const& array,
    U const& value)
{
    return std::move(MaskedArray<U, 2>(
        extents[array.shape()[0]][array.shape()[1]], value));
}


template<
    typename U,
    typename V>
inline MaskedArray<U, 3> clone(
    MaskedArray<V, 3> const& array)
{
    return std::move(MaskedArray<U, 3>(
        extents[array.shape()[0]][array.shape()[1]][array.shape()[1]]));
}


template<
    typename U,
    typename V>
inline MaskedArray<U, 3> clone(
    MaskedArray<V, 3> const& array,
    U const& value)
{
    return std::move(MaskedArray<U, 3>(
        extents[array.shape()[0]][array.shape()[1]][array.shape()[2]], value));
}

} // namespace fern

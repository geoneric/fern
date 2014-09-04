#pragma once
#include "fern/feature/core/array_traits.h"
#include "fern/feature/core/masked_array.h"


namespace fern {

template<
    class T,
    size_t nr_dimensions>
struct ArgumentTraits<
    MaskedArray<T, nr_dimensions>>
{

    using argument_category = typename detail::dispatch::ArrayCategoryTag<T,
        nr_dimensions>::type;

    template<
        class U>
    struct Collection
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
    class T,
    size_t nr_dimensions>
inline size_t size(
    MaskedArray<T, nr_dimensions> const& array)
{
    return array.num_elements();
}


template<
    class T,
    size_t nr_dimensions>
inline size_t size(
    MaskedArray<T, nr_dimensions> const& array,
    size_t dimension)
{
    assert(dimension < array.num_dimensions());
    return array.shape()[dimension];
}


template<
    class T>
inline typename ArgumentTraits<MaskedArray<T, 1>>::const_reference get(
    MaskedArray<T, 1> const& array,
    size_t index)
{
    assert(index < array.shape()[0]);
    // Don't assert this. Depending on the policy used, mask may not be
    // relevant.
    // assert(!array.mask()[index]);
    return array[index];
}


template<
    class T>
inline typename ArgumentTraits<MaskedArray<T, 1>>::reference get(
    MaskedArray<T, 1>& array,
    size_t index)
{
    assert(index < array.shape()[0]);
    // assert(!array.mask()[index]);
    return array[index];
}


template<
    class U,
    class V>
inline MaskedArray<U, 1> clone(
    MaskedArray<V, 1> const& array)
{
    return std::move(MaskedArray<U, 1>(extents[array.shape()[0]]));
}


template<
    class U,
    class V>
inline MaskedArray<U, 1> clone(
    MaskedArray<V, 1> const& array,
    U const& value)
{
    return std::move(MaskedArray<U, 1>(extents[array.shape()[0]], value));
}


template<
    class T>
inline typename ArgumentTraits<MaskedArray<T, 2>>::const_reference get(
    MaskedArray<T, 2> const& array,
    size_t index1,
    size_t index2)
{
    assert(index1 < array.shape()[0]);
    assert(index2 < array.shape()[1]);
    // assert(!array.mask()[index1][index2]);
    return array[index1][index2];
}


template<
    class T>
inline typename ArgumentTraits<MaskedArray<T, 2>>::reference get(
    MaskedArray<T, 2>& array,
    size_t index1,
    size_t index2)
{
    assert(index1 < array.shape()[0]);
    assert(index2 < array.shape()[1]);
    // assert(!array.mask()[index1][index2]);
    return array[index1][index2];
}


template<
    class U,
    class V>
inline MaskedArray<U, 2> clone(
    MaskedArray<V, 2> const& array)
{
    return std::move(MaskedArray<U, 2>(
        extents[array.shape()[0]][array.shape()[1]]));
}


template<
    class U,
    class V>
inline MaskedArray<U, 2> clone(
    MaskedArray<V, 2> const& array,
    U const& value)
{
    return std::move(MaskedArray<U, 2>(
        extents[array.shape()[0]][array.shape()[1]], value));
}


template<
    class T>
inline typename ArgumentTraits<MaskedArray<T, 3>>::const_reference get(
    MaskedArray<T, 3> const& array,
    size_t index1,
    size_t index2,
    size_t index3)
{
    assert(index1 < array.shape()[0]);
    assert(index2 < array.shape()[1]);
    assert(index3 < array.shape()[2]);
    // assert(!array.mask()[index1][index2][index3]);
    return array[index1][index2][index3];
}


template<
    class T>
inline typename ArgumentTraits<MaskedArray<T, 3>>::reference get(
    MaskedArray<T, 3>& array,
    size_t index1,
    size_t index2,
    size_t index3)
{
    assert(index1 < array.shape()[0]);
    assert(index2 < array.shape()[1]);
    assert(index3 < array.shape()[2]);
    // assert(!array.mask()[index1][index2][index3]);
    return array[index1][index2][index3];
}


template<
    class U,
    class V>
inline MaskedArray<U, 3> clone(
    MaskedArray<V, 3> const& array)
{
    return std::move(MaskedArray<U, 3>(
        extents[array.shape()[0]][array.shape()[1]][array.shape()[1]]));
}


template<
    class U,
    class V>
inline MaskedArray<U, 3> clone(
    MaskedArray<V, 3> const& array,
    U const& value)
{
    return std::move(MaskedArray<U, 3>(
        extents[array.shape()[0]][array.shape()[1]][array.shape()[2]], value));
}

} // namespace fern

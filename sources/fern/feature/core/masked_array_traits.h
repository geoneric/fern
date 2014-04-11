#pragma once
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

} // namespace fern

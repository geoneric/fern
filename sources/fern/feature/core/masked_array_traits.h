#pragma once
#include "fern/core/argument_traits.h"
#include "fern/feature/core/masked_array.h"


namespace fern {

template<
    class T,
    size_t nr_dimensions>
struct ArgumentTraits<
    MaskedArray<T, nr_dimensions>>
{

    typedef collection_tag argument_category;

    template<
        class U>
    struct Collection
    {
        typedef MaskedArray<T, nr_dimensions> type;
    };

    typedef T value_type;

    typedef typename MaskedArray<T, nr_dimensions>::element const*
        const_iterator;

    typedef typename MaskedArray<T, nr_dimensions>::element* iterator;

};


template<
    class T,
    size_t nr_dimensions>
typename ArgumentTraits<MaskedArray<T, nr_dimensions>>::const_iterator begin(
    MaskedArray<T, nr_dimensions> const& array)
{
    return array.data();
}


template<
    class T,
    size_t nr_dimensions>
typename ArgumentTraits<MaskedArray<T, nr_dimensions>>::iterator begin(
    MaskedArray<T, nr_dimensions>& array)
{
    return array.data();
}


template<
    class T,
    size_t nr_dimensions>
typename ArgumentTraits<MaskedArray<T, nr_dimensions>>::const_iterator end(
    MaskedArray<T, nr_dimensions> const& array)
{
    return array.data() + array.num_elements();
}


template<
    class T,
    size_t nr_dimensions>
typename ArgumentTraits<MaskedArray<T, nr_dimensions>>::iterator end(
    MaskedArray<T, nr_dimensions>& array)
{
    return array.end() + array.num_elements();
}


// template<
//     class U,
//     class V,
//     size_t nr_dimensions>
// void resize(
//     MaskedArray<U, nr_dimensions>& array,
//     MaskedArray<V, nr_dimensions> const& other_array)
// {
//     // array.resize(other_array.shape());
//     static_assert(nr_dimensions == 2, "");
//     array.resize(extents[other_array.shape()[0]][other_array.shape()[1]]);
// }


template<
    class T,
    size_t nr_dimensions>
size_t size(
    MaskedArray<T, nr_dimensions> const& array)
{
    return array.num_elements();
}


template<
    class T,
    size_t nr_dimensions>
T const& get(
    MaskedArray<T, nr_dimensions> const& array,
    size_t index)
{
    return array.data()[index];
}


template<
    class T,
    size_t nr_dimensions>
T& get(
    MaskedArray<T, nr_dimensions>& array,
    size_t index)
{
    return array.data()[index];
}

} // namespace fern

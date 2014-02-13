#pragma once
#include "fern/core/argument_traits.h"
#include "fern/feature/core/array.h"


namespace fern {

template<
    class T,
    size_t nr_dimensions>
struct ArgumentTraits<
    Array<T, nr_dimensions>>
{

    typedef collection_tag argument_category;

    template<
        class U>
    struct Collection
    {
        typedef Array<T, nr_dimensions> type;
    };

    typedef T value_type;

    typedef typename Array<T, nr_dimensions>::element const* const_iterator;

    typedef typename Array<T, nr_dimensions>::element* iterator;

};


template<
    class T,
    size_t nr_dimensions>
size_t size(
    Array<T, nr_dimensions> const& array)
{
    return array.num_elements();
}


template<
    class T,
    size_t nr_dimensions>
T const& get(
    Array<T, nr_dimensions> const& array,
    size_t index)
{
    return array.data()[index];
}


template<
    class T,
    size_t nr_dimensions>
T& get(
    Array<T, nr_dimensions>& array,
    size_t index)
{
    return array.data()[index];
}


// template<
//     class T,
//     size_t nr_dimensions>
// typename ArgumentTraits<Array<T, nr_dimensions>>::const_iterator begin(
//     Array<T, nr_dimensions> const& array)
// {
//     return array.data();
// }
// 
// 
// template<
//     class T,
//     size_t nr_dimensions>
// typename ArgumentTraits<Array<T, nr_dimensions>>::iterator begin(
//     Array<T, nr_dimensions>& array)
// {
//     return array.data();
// }
// 
// 
// template<
//     class T,
//     size_t nr_dimensions>
// typename ArgumentTraits<Array<T, nr_dimensions>>::const_iterator end(
//     Array<T, nr_dimensions> const& array)
// {
//     return array.data() + array.num_elements();
// }
// 
// 
// template<
//     class T,
//     size_t nr_dimensions>
// typename ArgumentTraits<Array<T, nr_dimensions>>::iterator end(
//     Array<T, nr_dimensions>& array)
// {
//     return array.end() + array.num_elements();
// }
// 
// 
// template<
//     class U,
//     class V,
//     size_t nr_dimensions>
// void resize(
//     Array<U, nr_dimensions>& array,
//     Array<V, nr_dimensions> const& other_array)
// {
//     // array.resize(other_array.shape());
//     static_assert(nr_dimensions == 2, "");
//     array.resize(extents[other_array.shape()[0]][other_array.shape()[1]]);
// }

} // namespace fern

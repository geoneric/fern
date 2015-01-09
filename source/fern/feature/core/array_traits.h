#pragma once
#include <cassert>
#include "fern/core/argument_traits.h"
#include "fern/feature/core/array.h"


namespace fern {
namespace detail {
namespace dispatch {

template<
    typename T,
    size_t nr_dimensions>
struct ArrayCategoryTag
{
};


#define ARRAY_CATEGORY_TAG(                    \
    nr_dimensions)                             \
template<                                      \
    typename T>                                \
struct ArrayCategoryTag<T, nr_dimensions>      \
{                                              \
                                               \
    using type = array_##nr_dimensions##d_tag; \
                                               \
};

ARRAY_CATEGORY_TAG(1)
ARRAY_CATEGORY_TAG(2)
ARRAY_CATEGORY_TAG(3)

#undef ARRAY_CATEGORY_TAG

} // namespace dispatch
} // namespace detail


// template<
//     typename T,
//     size_t nr_dimensions>
// struct ArgumentTraits<
//     View<T, nr_dimensions>>
// {
// 
//     using argument_category = typename detail::dispatch::ArrayCategoryTag<T, nr_dimensions>::type;
// 
//     template<
//         typename U>
//     struct Collection
//     {
//         using type = Array<T, nr_dimensions>;
//     };
// 
//     using value_type = T;
// 
// };


template<
    typename T,
    size_t nr_dimensions>
struct ArgumentTraits<
    Array<T, nr_dimensions>>
{

    using argument_category = typename detail::dispatch::ArrayCategoryTag<T,
        nr_dimensions>::type;

    template<
        typename U>
    struct Collection
    {
        using type = Array<U, nr_dimensions>;
    };

    template<
        typename U>
    struct Clone
    {
        using type = Array<U, nr_dimensions>;
    };

    using value_type = T;

    using reference = T&;

    using const_reference = T const&;

    static bool const is_masking = false;

    static size_t const rank = nr_dimensions;

};


template<
    typename T,
    size_t nr_dimensions>
inline size_t size(
    Array<T, nr_dimensions> const& array)
{
    return array.num_elements();
}


template<
    typename T,
    size_t nr_dimensions>
inline size_t size(
    Array<T, nr_dimensions> const& array,
    size_t dimension)
{
    assert(dimension < array.num_dimensions());
    return array.shape()[dimension];
}


template<
    typename T,
    size_t nr_dimensions>
inline size_t index(
    Array<T, nr_dimensions> const& array,
    size_t index1,
    size_t index2)
{
    return index1 * size(array, 1) + index2;
}


template<
    typename T,
    size_t nr_dimensions>
inline typename ArgumentTraits<Array<T, nr_dimensions>>::const_reference get(
    Array<T, nr_dimensions> const& array,
    size_t index)
{
    assert(index < array.num_elements());
    return array.data()[index];
}


template<
    typename T,
    size_t nr_dimensions>
inline typename ArgumentTraits<Array<T, nr_dimensions>>::reference get(
    Array<T, nr_dimensions>& array,
    size_t index)
{
    assert(index < array.num_elements());
    return array.data()[index];
}


template<
    typename U,
    typename V>
inline Array<U, 1> clone(
    Array<V, 1> const& array)
{
    return std::move(Array<U, 1>(extents[array.shape()[0]]));
}


template<
    typename U,
    typename V>
inline Array<U, 1> clone(
    Array<V, 1> const& array,
    U const& value)
{
    return std::move(Array<U, 1>(extents[array.shape()[0]], value));
}


template<
    typename U,
    typename V>
inline Array<U, 2> clone(
    Array<V, 2> const& array)
{
    return std::move(Array<U, 2>(extents[array.shape()[0]][array.shape()[1]]));
}


template<
    typename U,
    typename V>
inline Array<U, 2> clone(
    Array<V, 2> const& array,
    U const& value)
{
    return std::move(Array<U, 2>(extents[array.shape()[0]][array.shape()[1]],
        value));
}


template<
    typename U,
    typename V>
inline Array<U, 3> clone(
    Array<V, 3> const& array)
{
    return std::move(Array<U, 3>(
        extents[array.shape()[0]][array.shape()[1]][array.shape()[2]]));
}


template<
    typename U,
    typename V>
inline Array<U, 3> clone(
    Array<V, 3> const& array,
    U const& value)
{
    return std::move(Array<U, 3>(
        extents[array.shape()[0]][array.shape()[1]][array.shape()[2]],
        value));
}

} // namespace fern

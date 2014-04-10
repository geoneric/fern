#pragma once
#include "fern/core/argument_traits.h"
#include "fern/feature/core/array.h"


namespace fern {
namespace detail {
namespace dispatch {

template<
    class T,
    size_t nr_dimensions>
struct ArrayCategoryTag
{
};


#define ARRAY_CATEGORY_TAG(                    \
    nr_dimensions)                             \
template<                                      \
    class T>                                   \
struct ArrayCategoryTag<T, nr_dimensions>      \
{                                              \
                                               \
    typedef array_##nr_dimensions##d_tag type; \
                                               \
};

ARRAY_CATEGORY_TAG(1)
ARRAY_CATEGORY_TAG(2)
ARRAY_CATEGORY_TAG(3)

#undef ARRAY_CATEGORY_TAG

} // namespace dispatch
} // namespace detail


// template<
//     class T,
//     size_t nr_dimensions>
// struct ArgumentTraits<
//     View<T, nr_dimensions>>
// {
// 
//     typedef typename detail::dispatch::ArrayCategoryTag<T, nr_dimensions>::type
//         argument_category;
// 
//     template<
//         class U>
//     struct Collection
//     {
//         typedef Array<T, nr_dimensions> type;
//     };
// 
//     typedef T value_type;
// 
// };


template<
    class T,
    size_t nr_dimensions>
struct ArgumentTraits<
    Array<T, nr_dimensions>>
{

    typedef typename detail::dispatch::ArrayCategoryTag<T, nr_dimensions>::type
        argument_category;

    template<
        class U>
    struct Collection
    {
        typedef Array<U, nr_dimensions> type;
    };

    typedef T value_type;

    static bool const is_masking = false;

};


template<
    class T,
    size_t nr_dimensions>
inline size_t size(
    Array<T, nr_dimensions> const& array)
{
    return array.num_elements();
}


template<
    class T,
    size_t nr_dimensions>
inline size_t size(
    Array<T, nr_dimensions> const& array,
    size_t dimension)
{
    assert(dimension < array.num_dimensions());
    return array.shape()[dimension];
}


template<
    class T>
inline T const& get(
    Array<T, 1> const& array,
    size_t index)
{
    assert(index < array.shape()[0]);
    return array[index];
}


template<
    class T>
inline T& get(
    Array<T, 1>& array,
    size_t index)
{
    assert(index < array.shape()[0]);
    return array[index];
}


template<
    class T>
inline T const& get(
    Array<T, 2> const& array,
    size_t index1,
    size_t index2)
{
    assert(index1 < array.shape()[0]);
    assert(index2 < array.shape()[1]);
    return array[index1][index2];
}


template<
    class T>
inline T& get(
    Array<T, 2>& array,
    size_t index1,
    size_t index2)
{
    assert(index1 < array.shape()[0]);
    assert(index2 < array.shape()[1]);
    return array[index1][index2];
}


template<
    class T>
inline T const& get(
    Array<T, 3> const& array,
    size_t index1,
    size_t index2,
    size_t index3)
{
    assert(index1 < array.shape()[0]);
    assert(index2 < array.shape()[1]);
    assert(index3 < array.shape()[2]);
    return array[index1][index2][index3];
}


template<
    class T>
inline T& get(
    Array<T, 3>& array,
    size_t index1,
    size_t index2,
    size_t index3)
{
    assert(index1 < array.shape()[0]);
    assert(index2 < array.shape()[1]);
    assert(index3 < array.shape()[2]);
    return array[index1][index2][index3];
}

} // namespace fern

#pragma once
#include "fern/feature/core/array_traits.h"
#include "fern/feature/core/array_view.h"


namespace fern {

template<
    class T,
    size_t nr_dimensions>
struct ArgumentTraits<
    ArrayView<T, nr_dimensions>>
{

    using argument_category = typename detail::dispatch::ArrayCategoryTag<T,
        nr_dimensions>::type argument_category;

    template<
        class U>
    struct Collection
    {
        using type = ArrayView<U, nr_dimensions>;
    };

    using value_type = T;

    static bool const is_masking = false;

};


template<
    class T,
    size_t nr_dimensions>
size_t size(
    ArrayView<T, nr_dimensions> const& view)
{
    return view.num_elements();
}


template<
    class T,
    size_t nr_dimensions>
size_t size(
    ArrayView<T, nr_dimensions> const& view,
    size_t dimension)
{
    assert(dimension < view.num_dimensions());
    return view.shape()[dimension];
}


template<
    class T>
T const& get(
    ArrayView<T, 1> const& view,
    size_t index)
{
    assert(index < view.shape()[0]);
    return view[index];
}


template<
    class T>
T& get(
    ArrayView<T, 1>& view,
    size_t index)
{
    assert(index < view.shape()[0]);
    return view[index];
}


template<
    class T>
T const& get(
    ArrayView<T, 2> const& view,
    size_t index1,
    size_t index2)
{
    assert(index1 < view.shape()[0]);
    assert(index2 < view.shape()[1]);
    return view[index1][index2];
}


template<
    class T>
T& get(
    ArrayView<T, 2>& view,
    size_t index1,
    size_t index2)
{
    assert(index1 < view.shape()[0]);
    assert(index2 < view.shape()[1]);
    return view[index1][index2];
}


template<
    class T>
T const& get(
    ArrayView<T, 3> const& view,
    size_t index1,
    size_t index2,
    size_t index3)
{
    assert(index1 < view.shape()[0]);
    assert(index2 < view.shape()[1]);
    assert(index3 < view.shape()[2]);
    return view[index1][index2][index3];
}


template<
    class T>
T& get(
    ArrayView<T, 3>& view,
    size_t index1,
    size_t index2,
    size_t index3)
{
    assert(index1 < view.shape()[0]);
    assert(index2 < view.shape()[1]);
    assert(index3 < view.shape()[2]);
    return view[index1][index2][index3];
}

} // namespace fern

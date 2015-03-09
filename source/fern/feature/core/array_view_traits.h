#pragma once
#include "fern/feature/core/array_traits.h"
#include "fern/feature/core/array_view.h"


namespace fern {

template<
    typename T,
    size_t nr_dimensions>
struct DataTraits<
    ArrayView<T, nr_dimensions>>
{

    using argument_category = typename detail::dispatch::ArrayCategoryTag<T,
        nr_dimensions>::type;

    template<
        typename U>
    struct Collection
    {
        using type = ArrayView<U, nr_dimensions>;
    };

    using value_type = T;

    static bool const is_masking = false;

};


template<
    typename T,
    size_t nr_dimensions>
size_t size(
    ArrayView<T, nr_dimensions> const& view)
{
    return view.num_elements();
}


template<
    typename T,
    size_t nr_dimensions>
size_t size(
    ArrayView<T, nr_dimensions> const& view,
    size_t dimension)
{
    assert(dimension < view.num_dimensions());
    return view.shape()[dimension];
}


template<
    typename T,
    size_t nr_dimensions>
T const& get(
    ArrayView<T, nr_dimensions> const& view,
    size_t index)
{
    assert(index < view.num_elements());
    return view.data()[index];
}


template<
    typename T,
    size_t nr_dimensions>
T& get(
    ArrayView<T, nr_dimensions>& view,
    size_t index)
{
    assert(index < view.num_elements());
    return view.data()[index];
}

} // namespace fern

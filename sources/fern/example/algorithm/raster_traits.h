#pragma once
#include <cassert>
#include "fern/core/argument_traits.h"
#include "fern/example/algorithm/raster.h"


namespace fern {

template<
    class T>
struct ArgumentTraits<example::Raster<T>>
{

    template<
        class U>
    struct Collection
    {
        using type = example::Raster<U>;
    };

    using value_type = T;

    using const_reference = T const&;

    using reference = T&;

    using argument_category = array_2d_tag;

};


template<
    class T>
size_t size(
    example::Raster<T> const& raster,
    size_t index)
{
    assert(index == 0 || index == 1);
    return index == 0 ? raster.nr_rows() : raster.nr_cols();
}


template<
    class T>
T const& get(
    example::Raster<T> const& raster,
    size_t row,
    size_t col)
{
    return raster.values()[row * raster.nr_cols() + col];
}


template<
    class T>
T& get(
    example::Raster<T>& raster,
    size_t row,
    size_t col)
{
    return raster.values()[row * raster.nr_cols() + col];
}


template<
    class T>
double cell_size(
    example::Raster<T> const& raster,
    size_t /* index */)
{
    return raster.cell_size();
}

} // namespace fern

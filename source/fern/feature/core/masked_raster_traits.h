#pragma once
#include <cassert>
#include "fern/feature/core/raster_traits.h"
#include "fern/feature/core/masked_raster.h"


namespace fern {

template<
    typename T,
    size_t nr_dimensions>
struct DataTraits<
    MaskedRaster<T, nr_dimensions>>
{

    using argument_category = typename detail::dispatch::RasterCategoryTag<T,
        nr_dimensions>::type;

    template<
        typename U>
    struct Collection
    {
        using type = MaskedRaster<U, nr_dimensions>;
    };

    template<
        typename U>
    struct Clone
    {
        using type = MaskedRaster<U, nr_dimensions>;
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
    MaskedRaster<T, nr_dimensions> const& raster)
{
    return raster.num_elements();
}


template<
    typename T,
    size_t nr_dimensions>
inline size_t size(
    MaskedRaster<T, nr_dimensions> const& raster,
    size_t dimension)
{
    assert(dimension < raster.num_dimensions());
    return raster.shape()[dimension];
}


template<
    typename T,
    size_t nr_dimensions>
inline size_t index(
    MaskedRaster<T, nr_dimensions> const& raster,
    size_t index1,
    size_t index2)
{
    return index1 * size(raster, 1) + index2;
}


template<
    typename T,
    size_t nr_dimensions>
inline typename DataTraits<MaskedRaster<T, nr_dimensions>>
        ::const_reference get(
    MaskedRaster<T, nr_dimensions> const& raster,
    size_t index)
{
    assert(index < raster.num_elements());
    return raster.data()[index];
}


template<
    typename T,
    size_t nr_dimensions>
inline typename DataTraits<MaskedRaster<T, nr_dimensions>>::reference get(
    MaskedRaster<T, nr_dimensions>& raster,
    size_t index)
{
    assert(index < raster.num_elements());
    return raster.data()[index];
}


template<
    typename U,
    typename V>
inline MaskedRaster<U, 1> clone(
    MaskedRaster<V, 1> const& raster)
{
    return std::move(MaskedRaster<U, 1>(extents[raster.shape()[0]],
        raster.transformation()));
}


template<
    typename U,
    typename V>
inline MaskedRaster<U, 1> clone(
    MaskedRaster<V, 1> const& raster,
    U const& value)
{
    return std::move(MaskedRaster<U, 1>(extents[raster.shape()[0]],
        raster.transformation(),
        value));
}


template<
    typename T>
inline double cell_size(
    MaskedRaster<T, 2> const& raster,
    size_t index)
{
    assert(index == 0 || index == 1);
    return raster.transformation()[index == 0 ? 1 : 3];
}


template<
    typename T>
inline double cell_area(
    MaskedRaster<T, 2> const& raster)
{
    return raster.transformation()[1] * raster.transformation()[3];
}


template<
    typename U,
    typename V>
inline MaskedRaster<U, 2> clone(
    MaskedRaster<V, 2> const& raster)
{
    return std::move(MaskedRaster<U, 2>(
        extents[raster.shape()[0]][raster.shape()[1]],
        raster.transformation()));
}


template<
    typename U,
    typename V>
inline MaskedRaster<U, 2> clone(
    MaskedRaster<V, 2> const& raster,
    U const& value)
{
    return std::move(MaskedRaster<U, 2>(
        extents[raster.shape()[0]][raster.shape()[1]],
        raster.transformation(),
        value));
}


template<
    typename U,
    typename V>
inline MaskedRaster<U, 3> clone(
    MaskedRaster<V, 3> const& raster)
{
    return std::move(MaskedRaster<U, 3>(
        extents[raster.shape()[0]][raster.shape()[1],raster.shape()[2]],
        raster.transformation()));
}


template<
    typename U,
    typename V>
inline MaskedRaster<U, 3> clone(
    MaskedRaster<V, 3> const& raster,
    U const& value)
{
    return std::move(MaskedRaster<U, 3>(
        extents[raster.shape()[0]][raster.shape()[1],raster.shape()[2]],
        raster.transformation(),
        value));
}

} // namespace fern

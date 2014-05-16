#pragma once
#include "fern/feature/core/raster_traits.h"
#include "fern/feature/core/masked_raster.h"


namespace fern {

template<
    class T,
    size_t nr_dimensions>
struct ArgumentTraits<
    MaskedRaster<T, nr_dimensions>>
{

    using argument_category = typename detail::dispatch::RasterCategoryTag<T,
        nr_dimensions>::type;

    template<
        class U>
    struct Collection
    {
        using type = MaskedRaster<U, nr_dimensions>;
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
    MaskedRaster<T, nr_dimensions> const& raster)
{
    return raster.num_elements();
}


template<
    class T,
    size_t nr_dimensions>
inline size_t size(
    MaskedRaster<T, nr_dimensions> const& raster,
    size_t dimension)
{
    assert(dimension < raster.num_dimensions());
    return raster.shape()[dimension];
}


template<
    class T>
inline typename ArgumentTraits<MaskedRaster<T, 1>>::const_reference get(
    MaskedRaster<T, 1> const& raster,
    size_t index)
{
    assert(index < raster.shape()[0]);
    // Don't assert this. Depending on the policy used, mask may not be
    // relevant.
    // assert(!raster.mask()[index]);
    return raster[index];
}


template<
    class T>
inline typename ArgumentTraits<MaskedRaster<T, 1>>::reference get(
    MaskedRaster<T, 1>& raster,
    size_t index)
{
    assert(index < raster.shape()[0]);
    // assert(!raster.mask()[index]);
    return raster[index];
}


template<
    class T>
inline typename ArgumentTraits<MaskedRaster<T, 2>>::const_reference get(
    MaskedRaster<T, 2> const& raster,
    size_t index1,
    size_t index2)
{
    assert(index1 < raster.shape()[0]);
    assert(index2 < raster.shape()[1]);
    // assert(!raster.mask()[index1][index2]);
    return raster[index1][index2];
}


template<
    class T>
inline typename ArgumentTraits<MaskedRaster<T, 2>>::reference get(
    MaskedRaster<T, 2>& raster,
    size_t index1,
    size_t index2)
{
    assert(index1 < raster.shape()[0]);
    assert(index2 < raster.shape()[1]);
    // assert(!raster.mask()[index1][index2]);
    return raster[index1][index2];
}


template<
    class T>
inline typename ArgumentTraits<MaskedRaster<T, 3>>::const_reference get(
    MaskedRaster<T, 3> const& raster,
    size_t index1,
    size_t index2,
    size_t index3)
{
    assert(index1 < raster.shape()[0]);
    assert(index2 < raster.shape()[1]);
    assert(index3 < raster.shape()[2]);
    // assert(!raster.mask()[index1][index2][index3]);
    return raster[index1][index2][index3];
}


template<
    class T>
inline typename ArgumentTraits<MaskedRaster<T, 3>>::reference get(
    MaskedRaster<T, 3>& raster,
    size_t index1,
    size_t index2,
    size_t index3)
{
    assert(index1 < raster.shape()[0]);
    assert(index2 < raster.shape()[1]);
    assert(index3 < raster.shape()[2]);
    // assert(!raster.mask()[index1][index2][index3]);
    return raster[index1][index2][index3];
}

} // namespace fern

#pragma once
#include <cassert>
#include "fern/core/data_customization_point.h"
#include "fern/python/feature/detail/data_traits/masked_raster.h"


namespace fern {

template<
    typename T>
inline size_t size(
    python::detail::MaskedRaster<T> const& raster)
{
    return raster.size();
}


template<
    typename T>
inline size_t size(
    python::detail::MaskedRaster<T> const& raster,
    size_t dimension)
{
    assert(dimension == 0 || dimension == 1);
    return dimension == 0
        ? std::get<0>(raster.sizes())
        : std::get<1>(raster.sizes())
        ;
}


template<
    typename T>
inline double cell_size(
    python::detail::MaskedRaster<T> const& raster,
    size_t index)
{
    assert(index == 0 || index == 1);
    return index == 0
        ? std::get<0>(raster.cell_sizes())
        : std::get<1>(raster.cell_sizes())
        ;
}


template<
    typename T>
inline double cell_area(
    python::detail::MaskedRaster<T> const& raster)
{
    return raster.cell_area();
}


template<
    typename T>
inline size_t index(
    python::detail::MaskedRaster<T> const& raster,
    size_t index1,
    size_t index2)
{
    return raster.index(index1, index2);
}


template<
    typename T>
inline typename DataTraits<python::detail::MaskedRaster<T>>::const_reference get(
    python::detail::MaskedRaster<T> const& raster,
    size_t index)
{
    assert(index < size(raster));
    return raster.element(index);
}


template<
    typename T>
inline typename DataTraits<python::detail::MaskedRaster<T>>::reference get(
    python::detail::MaskedRaster<T>& raster,
    size_t index)
{
    assert(index < size(raster));
    return raster.element(index);
}


template<
    typename T>
inline typename DataTraits<python::detail::MaskedRaster<T>>::const_reference get(
    python::detail::MaskedRaster<T> const& raster,
    size_t index1,
    size_t index2)
{
    assert(index1 < size(raster, 0));
    assert(index2 < size(raster, 1));
    return raster.element(index1, index2);
}


template<
    typename T>
inline typename DataTraits<python::detail::MaskedRaster<T>>::reference get(
    python::detail::MaskedRaster<T>& raster,
    size_t index1,
    size_t index2)
{
    assert(index1 < size(raster, 0));
    assert(index2 < size(raster, 1));
    return raster.element(index1, index2);
}


template<
    typename U,
    typename V>
inline python::detail::MaskedRaster<U> clone(
    python::detail::MaskedRaster<V> const& raster)
{
    return python::detail::MaskedRaster<U>(raster.sizes(), raster.origin(),
        raster.cell_sizes());
}


template<
    typename U,
    typename V>
inline python::detail::MaskedRaster<U> clone(
    python::detail::MaskedRaster<V> const& raster,
    U const& value)
{
    return python::detail::MaskedRaster<U>(raster.sizes(), raster.origin(),
        raster.cell_sizes(), value);
}

} // namespace fern

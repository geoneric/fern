// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include <cstddef>
#include <utility>
#include "fern/core/data_customization_point.h"
#include "fern/feature/core/data_type_traits/raster.h"


namespace fern {

template<
    typename T,
    size_t nr_dimensions>
inline size_t size(
    Raster<T, nr_dimensions> const& raster)
{
    return raster.num_elements();
}


template<
    typename T,
    size_t nr_dimensions>
inline size_t size(
    Raster<T, nr_dimensions> const& raster,
    size_t dimension)
{
    assert(dimension < raster.num_dimensions());
    return raster.shape()[dimension];
}


template<
    typename T,
    size_t nr_dimensions>
inline size_t index(
    Raster<T, nr_dimensions> const& raster,
    size_t index1,
    size_t index2)
{
    return index1 * size(raster, 1) + index2;
}


template<
    typename T,
    size_t nr_dimensions>
inline typename DataTypeTraits<Raster<T, nr_dimensions>>::const_reference get(
    Raster<T, nr_dimensions> const& raster,
    size_t index)
{
    assert(index < raster.num_elements());
    return raster.data()[index];
}


template<
    typename T,
    size_t nr_dimensions>
inline typename DataTypeTraits<Raster<T, nr_dimensions>>::reference get(
    Raster<T, nr_dimensions>& raster,
    size_t index)
{
    assert(index < raster.num_elements());
    return raster.data()[index];
}


template<
    typename U,
    typename V>
inline Raster<U, 1> clone(
    Raster<V, 1> const& raster)
{
    return Raster<U, 1>(extents[raster.shape()[0]],
        raster.transformation());
}


template<
    typename U,
    typename V>
inline Raster<U, 1> clone(
    Raster<V, 1> const& raster,
    U const& value)
{
    return Raster<U, 1>(extents[raster.shape()[0]], raster.transformation(),
        value);
}


template<
    typename T>
inline double cell_size(
    Raster<T, 2> const& raster,
    size_t index)
{
    assert(index == 0 || index == 1);
    return raster.transformation()[index == 0 ? 1 : 3];
}


template<
    typename T>
inline double cell_area(
    Raster<T, 2> const& raster)
{
    return raster.transformation()[1] * raster.transformation()[3];
}


template<
    typename U,
    typename V>
inline Raster<U, 2> clone(
    Raster<V, 2> const& raster)
{
    return Raster<U, 2>(extents[raster.shape()[0]][raster.shape()[1]],
        raster.transformation());
}


template<
    typename U,
    typename V>
inline Raster<U, 2> clone(
    Raster<V, 2> const& raster,
    U const& value)
{
    return Raster<U, 2>(extents[raster.shape()[0]][raster.shape()[1]],
        raster.transformation(), value);
}


template<
    typename U,
    typename V>
inline Raster<U, 3> clone(
    Raster<V, 3> const& raster)
{
    return Raster<U, 3>(
        extents[raster.shape()[0]][raster.shape()[1],raster.shape()[2]],
        raster.transformation());
}


template<
    typename U,
    typename V>
inline Raster<U, 3> clone(
    Raster<V, 3> const& raster,
    U const& value)
{
    return Raster<U, 3>(
        extents[raster.shape()[0]][raster.shape()[1],raster.shape()[2]],
        raster.transformation(), value);
}

} // namespace fern

// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include <cassert>
#include <utility>
#include "fern/core/data_customization_point.h"
#include "fern/feature/core/data_type_traits/array.h"


namespace fern {

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
inline typename DataTypeTraits<Array<T, nr_dimensions>>::const_reference get(
    Array<T, nr_dimensions> const& array,
    size_t index)
{
    assert(index < array.num_elements());
    return array.data()[index];
}


template<
    typename T,
    size_t nr_dimensions>
inline typename DataTypeTraits<Array<T, nr_dimensions>>::reference get(
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
    return Array<U, 1>(extents[array.shape()[0]]);
}


template<
    typename U,
    typename V>
inline Array<U, 1> clone(
    Array<V, 1> const& array,
    U const& value)
{
    return Array<U, 1>(extents[array.shape()[0]], value);
}


template<
    typename U,
    typename V>
inline Array<U, 2> clone(
    Array<V, 2> const& array)
{
    return Array<U, 2>(extents[array.shape()[0]][array.shape()[1]]);
}


template<
    typename U,
    typename V>
inline Array<U, 2> clone(
    Array<V, 2> const& array,
    U const& value)
{
    return Array<U, 2>(extents[array.shape()[0]][array.shape()[1]],
        value);
}


template<
    typename U,
    typename V>
inline Array<U, 3> clone(
    Array<V, 3> const& array)
{
    return Array<U, 3>(
        extents[array.shape()[0]][array.shape()[1]][array.shape()[2]]);
}


template<
    typename U,
    typename V>
inline Array<U, 3> clone(
    Array<V, 3> const& array,
    U const& value)
{
    return Array<U, 3>(
        extents[array.shape()[0]][array.shape()[1]][array.shape()[2]],
        value);
}

} // namespace fern

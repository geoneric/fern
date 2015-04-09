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
#include "fern/core/data_customization_point.h"
#include "fern/feature/core/data_traits/array_reference.h"


namespace fern {

template<
    class T,
    size_t nr_dimensions>
inline size_t size(
    ArrayReference<T, nr_dimensions> const& array)
{
    return array.num_elements();
}


template<
    class T,
    size_t nr_dimensions>
inline size_t size(
    ArrayReference<T, nr_dimensions> const& array,
    size_t dimension)
{
    assert(dimension < array.num_dimensions());
    return array.shape()[dimension];
}


template<
    class T,
    size_t nr_dimensions>
inline typename DataTraits<ArrayReference<T, nr_dimensions>>
        ::const_reference get(
    ArrayReference<T, nr_dimensions> const& array,
    size_t index)
{
    assert(index < array.num_elements());
    return array.data()[index];
}


template<
    class T,
    size_t nr_dimensions>
inline typename DataTraits<ArrayReference<T, nr_dimensions>>::reference get(
    ArrayReference<T, nr_dimensions>& array,
    size_t index)
{
    assert(index < array.num_elements());
    return array.data()[index];
}

} // namespace fern

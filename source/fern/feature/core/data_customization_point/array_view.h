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
#include "fern/feature/core/data_type_traits/array_view.h"


namespace fern {

template<
    typename T,
    size_t nr_dimensions>
inline size_t size(
    ArrayView<T, nr_dimensions> const& view)
{
    return view.num_elements();
}


template<
    typename T,
    size_t nr_dimensions>
inline size_t size(
    ArrayView<T, nr_dimensions> const& view,
    size_t dimension)
{
    assert(dimension < view.num_dimensions());
    return view.shape()[dimension];
}


template<
    typename T,
    size_t nr_dimensions>
inline typename DataTypeTraits<ArrayView<T, nr_dimensions>>::const_reference
        get(
    ArrayView<T, nr_dimensions> const& view,
    size_t index)
{
    assert(index < view.num_elements());
    return view.data()[index];
}


template<
    typename T,
    size_t nr_dimensions>
inline typename DataTypeTraits<ArrayView<T, nr_dimensions>>::reference get(
    ArrayView<T, nr_dimensions>& view,
    size_t index)
{
    assert(index < view.num_elements());
    return view.data()[index];
}

} // namespace fern

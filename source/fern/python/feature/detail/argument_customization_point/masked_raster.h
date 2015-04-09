// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include "fern/algorithm/core/argument_customization_point.h"
#include "fern/python/feature/detail/argument_traits/masked_raster.h"


namespace fern {
namespace algorithm {

template<
    typename T>
inline MaskT<python::detail::MaskedRaster<T>>& mask(
    python::detail::MaskedRaster<T>& argument)
{
    return argument;
}

} // namespace algorithm
} // namespace fern

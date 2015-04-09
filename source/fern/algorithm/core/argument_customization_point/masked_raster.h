// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include "fern/feature/core/masked_raster.h"
#include "fern/algorithm/core/argument_customization_point.h"


namespace fern {
namespace algorithm {

template<
    typename T,
    size_t nr_dimensions>
inline MaskT<MaskedRaster<T, nr_dimensions>>& mask(
    MaskedRaster<T, nr_dimensions>& argument)
{
    return argument.mask();
}

} // namespace algorithm
} // namespace fern

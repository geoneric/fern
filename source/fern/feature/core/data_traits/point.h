// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include "fern/core/data_traits.h"
#include "fern/feature/core/point.h"


namespace fern {

template<
    typename T,
    size_t nr_dimensions,
    typename CoordinateSystem>
struct DataTraits<
    Point<T, nr_dimensions, CoordinateSystem>>
{

    using value_type = T;

    static size_t const rank = nr_dimensions;

};

} // namespace fern

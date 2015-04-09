// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include "fern/algorithm/core/argument_traits.h"
#include "fern/algorithm/policy/detect_no_data.h"
#include "fern/algorithm/policy/mark_no_data.h"
#include "fern/python/feature/detail/masked_raster.h"


namespace fern {
namespace algorithm {

template<
    typename T>
struct ArgumentTraits<
    python::detail::MaskedRaster<T>>
{

    using Mask = python::detail::MaskedRaster<T>;

    using InputNoDataPolicy = DetectNoData<python::detail::MaskedRaster<T>>;

    using OutputNoDataPolicy = MarkNoData<python::detail::MaskedRaster<T>>;

};

} // namespace algorithm
} // namespace fern

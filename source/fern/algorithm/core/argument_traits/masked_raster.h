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
#include "fern/algorithm/policy/detect_no_data_by_value.h"
#include "fern/algorithm/policy/mark_no_data_by_value.h"
#include "fern/feature/core/masked_raster.h"


namespace fern {
namespace algorithm {

template<
    typename T,
    size_t nr_dimensions>
struct ArgumentTraits<
    MaskedRaster<T, nr_dimensions>>
{

    using Mask = fern::Mask<nr_dimensions>;

    using InputNoDataPolicy = DetectNoDataByValue<Mask>;

    using OutputNoDataPolicy = MarkNoDataByValue<Mask>;

};

} // namespace algorithm
} // namespace fern

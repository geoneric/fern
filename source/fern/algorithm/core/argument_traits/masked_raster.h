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

    using InputNoDataPolicy = algorithm::DetectNoDataByValue<Mask>;

    using OutputNoDataPolicy = algorithm::MarkNoDataByValue<Mask>;

};

} // namespace algorithm
} // namespace fern

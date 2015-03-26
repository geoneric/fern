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

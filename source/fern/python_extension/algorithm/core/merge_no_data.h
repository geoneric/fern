#pragma once
#include "fern/feature/core/array_traits.h"
#include "fern/feature/core/masked_raster_traits.h"
#include "fern/algorithm/policy/policies.h"
#include "fern/algorithm/core/merge_no_data.h"


namespace fern {
namespace python {

template<
    typename T,
    typename R>
inline void merge_no_data(
    algorithm::ExecutionPolicy& execution_policy,
    fern::MaskedRaster<T, 2> const& raster,
    fern::MaskedRaster<R, 2>& result_raster)
{
    using InputNoDataPolicy = algorithm::DetectNoDataByValue<fern::Mask<2>>;
    using OutputNoDataPolicy = algorithm::MarkNoDataByValue<fern::Mask<2>>;

    algorithm::SkipNoData<
        InputNoDataPolicy> input_no_data_policy(
            InputNoDataPolicy(raster.mask(), true));
    OutputNoDataPolicy output_no_data_policy(result_raster.mask(), true);

    algorithm::core::merge_no_data(input_no_data_policy,
        output_no_data_policy, execution_policy, raster, result_raster);
}

} // namespace python
} // namespace fern

#pragma once
#include "fern/feature/core/array_traits.h"
#include "fern/feature/core/masked_raster_traits.h"
#include "fern/algorithm/policy/policies.h"
#include "fern/algorithm/core/unite_no_data.h"


namespace fern {
namespace python {

template<
    typename T1,
    typename T2,
    typename R>
void unite_no_data(
    algorithm::ExecutionPolicy& execution_policy,
    fern::MaskedRaster<T1, 2> const& raster1,
    fern::MaskedRaster<T2, 2> const& raster2,
    fern::MaskedRaster<R, 2>& result_raster)
{
    using InputNoDataPolicy = algorithm::DetectNoDataByValue<fern::Mask<2>>;
    using OutputNoDataPolicy = algorithm::MarkNoDataByValue<fern::Mask<2>>;

    algorithm::SkipNoData<
        InputNoDataPolicy,
        InputNoDataPolicy> input_no_data_policy(
            InputNoDataPolicy(raster1.mask(), true),
            InputNoDataPolicy(raster2.mask(), true));
    OutputNoDataPolicy output_no_data_policy(result_raster.mask(), true);

    algorithm::core::unite_no_data(input_no_data_policy,
        output_no_data_policy, execution_policy, raster1, raster2,
        result_raster);
}

} // namespace python
} // namespace fern

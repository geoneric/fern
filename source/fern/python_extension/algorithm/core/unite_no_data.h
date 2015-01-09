#pragma once
#include "fern/python_extension/feature/detail/masked_raster_traits.h"
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
    detail::MaskedRaster<T1> const& raster1,
    detail::MaskedRaster<T2> const& raster2,
    detail::MaskedRaster<R>& result_raster)
{
    algorithm::SkipNoData<
        algorithm::DetectNoData<detail::MaskedRaster<T1>>,
        algorithm::DetectNoData<detail::MaskedRaster<T2>>>
        input_no_data_policy{raster1, raster2};
    algorithm::MarkNoData<detail::MaskedRaster<R>> output_no_data_policy{
        result_raster};

    algorithm::core::unite_no_data(input_no_data_policy,
        output_no_data_policy, execution_policy, raster1, raster2,
        result_raster);
}

} // namespace python
} // namespace fern

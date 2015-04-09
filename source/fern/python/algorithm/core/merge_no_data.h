// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include "fern/python/feature/detail/masked_raster_traits.h"
#include "fern/algorithm/policy/policies.h"
#include "fern/algorithm/core/merge_no_data.h"


namespace fern {
namespace python {

template<
    typename T,
    typename R>
inline void merge_no_data(
    algorithm::ExecutionPolicy& execution_policy,
    detail::MaskedRaster<T> const& raster,
    detail::MaskedRaster<R>& result_raster)
{
    algorithm::SkipNoData<
        algorithm::DetectNoData<detail::MaskedRaster<T>>>
        input_no_data_policy{raster};
    algorithm::MarkNoData<detail::MaskedRaster<R>> output_no_data_policy{
        result_raster};

    algorithm::core::merge_no_data(input_no_data_policy,
        output_no_data_policy, execution_policy, raster, result_raster);
}

} // namespace python
} // namespace fern

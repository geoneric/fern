#include "fern/python_extension/algorithm/space/focal/slope.h"
#include "fern/feature/core/array_traits.h"
#include "fern/feature/core/masked_raster_traits.h"
#include "fern/algorithm/policy/policies.h"
#include "fern/algorithm/space/focal/slope.h"
#include "fern/python_extension/core/switch_on_value_type.h"
#include "fern/python_extension/algorithm/core/merge_no_data.h"


namespace fa = fern::algorithm;

namespace fern {
namespace python {

template<
    typename T>
MaskedRasterHandle slope(
    fa::ExecutionPolicy& execution_policy,
    fern::MaskedRaster<T, 2> const& dem)
{
    using InputNoDataPolicy = algorithm::DetectNoDataByValue<Mask<2>>;
    using OutputNoDataPolicy = algorithm::MarkNoDataByValue<Mask<2>>;

    auto result_ptr(std::make_shared<fern::MaskedRaster<T, 2>>(
        extents[dem.shape()[0]][dem.shape()[1]], dem.transformation()));
    auto& result(*result_ptr);

    InputNoDataPolicy input_no_data_policy(dem.mask(), true);
    OutputNoDataPolicy output_no_data_policy(result.mask(), true);

    algorithm::space::slope<algorithm::slope::OutOfRangePolicy>(
        input_no_data_policy, output_no_data_policy, execution_policy,
        dem, result);

    return std::make_shared<MaskedRaster>(result_ptr);
}


#define CASE(                        \
    value_type_enum,                 \
    value_type)                      \
case value_type_enum: {              \
    result = slope<>(                \
        execution_policy,            \
        dem->raster<value_type>());  \
    break;                           \
}

MaskedRasterHandle slope(
    fa::ExecutionPolicy& execution_policy,
    MaskedRasterHandle const& dem)
{
    assert(!PyErr_Occurred());
    MaskedRasterHandle result;

    SWITCH_ON_FLOATING_POINT_VALUE_TYPE(dem->value_type(), CASE)

    assert(!PyErr_Occurred());
    return result;
}

#undef CASE

} // namespace python
} // namespace fern

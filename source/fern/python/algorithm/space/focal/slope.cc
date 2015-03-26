#include "fern/python/algorithm/space/focal/slope.h"
#include "fern/python/feature/detail/data_customization_point/masked_raster.h"
#include "fern/python/feature/detail/argument_customization_point/masked_raster.h"
#include "fern/algorithm/policy/policies.h"
#include "fern/algorithm/space/focal/slope.h"
#include "fern/python/core/switch_on_value_type.h"


namespace fa = fern::algorithm;

namespace fern {
namespace python {
namespace {

template<
    typename T>
MaskedRasterHandle slope(
    fa::ExecutionPolicy& execution_policy,
    detail::MaskedRaster<T> const& dem)
{
    auto result_ptr(std::make_shared<detail::MaskedRaster<T>>(
        dem.sizes(), dem.origin(), dem.cell_sizes()));
    auto& result(*result_ptr);

    fa::InputNoDataPolicies<
        fa::DetectNoData<detail::MaskedRaster<T>>
    > input_no_data_policy{
        fa::DetectNoData<detail::MaskedRaster<T>>{dem}};
    fa::MarkNoData<detail::MaskedRaster<T>> output_no_data_policy{result};

    algorithm::space::slope<algorithm::slope::OutOfRangePolicy>(
        input_no_data_policy, output_no_data_policy, execution_policy,
        dem, result);

    return std::make_shared<MaskedRaster>(result_ptr);
}

} // Anonymous namespace


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

#include "fern/python/algorithm/core/if.h"
#include "fern/python/feature/detail/data_customization_point/masked_raster.h"
#include "fern/algorithm/policy/policies.h"
#include "fern/algorithm/core/if.h"
#include "fern/algorithm/algebra/elementary/add.h"
#include "fern/python/core/switch_on_value_type.h"


namespace fa = fern::algorithm;


namespace fern {
namespace python {
namespace {

template<
    typename T1,
    typename T2,
    typename T3,
    typename R>
void if_(
    fa::ExecutionPolicy& execution_policy,
    detail::MaskedRaster<T1> const& condition,
    detail::MaskedRaster<T2> const& true_value,
    detail::MaskedRaster<T3> const& false_value,
    detail::MaskedRaster<R>& result)
{
    fa::InputNoDataPolicies<
        fa::DetectNoData<detail::MaskedRaster<T1>>,
        fa::DetectNoData<detail::MaskedRaster<T2>>,
        fa::DetectNoData<detail::MaskedRaster<T3>>
    > input_no_data_policy{
        fa::DetectNoData<detail::MaskedRaster<T1>>{condition},
        fa::DetectNoData<detail::MaskedRaster<T2>>{true_value},
        fa::DetectNoData<detail::MaskedRaster<T3>>{false_value}};
    fa::MarkNoData<detail::MaskedRaster<R>> output_no_data_policy(result);

    fa::core::if_(input_no_data_policy, output_no_data_policy,
        execution_policy, condition, true_value, false_value, result);
}


template<
    typename T1,
    typename T2,
    typename T3>
MaskedRasterHandle if_(
    fa::ExecutionPolicy& execution_policy,
    detail::MaskedRaster<T1> const& condition,
    detail::MaskedRaster<T2> const& true_value,
    detail::MaskedRaster<T3> const& false_value)
{
    // TODO Assert raster properties.
    using R = algorithm::add::result_value_type<T2, T3>;
    auto handle = std::make_shared<detail::MaskedRaster<R>>(condition.sizes(),
        condition.origin(), condition.cell_sizes());
    if_(execution_policy, condition, true_value, false_value, *handle);
    return std::make_shared<MaskedRaster>(handle);
}

} // Anonymous namespace


#define CASE2(                                \
    value_type_enum2,                         \
    value_type2,                              \
    value_type1)                              \
case value_type_enum2: {                      \
    result = if_(                             \
        execution_policy,                     \
        condition->raster<value_type1>(),     \
        true_value->raster<value_type2>(),    \
        false_value->raster<value_type2>());  \
    break;                                    \
}

#define CASE1(              \
    value_type_enum1,       \
    value_type1,            \
    value_type_enum2)       \
case value_type_enum1: {    \
    SWITCH_ON_VALUE_TYPE2(  \
        value_type_enum2,   \
        CASE2,              \
        value_type1)        \
    break;                  \
}

MaskedRasterHandle if_(
    fa::ExecutionPolicy& execution_policy,
    MaskedRasterHandle const& condition,
    MaskedRasterHandle const& true_value,
    MaskedRasterHandle const& false_value)
{
    if(true_value->value_type() != false_value->value_type()) {
        // TODO
        assert(false);
    }

    MaskedRasterHandle result;
    SWITCH_ON_VALUE_TYPE1(condition->value_type(), CASE1,
        true_value->value_type())
    return result;
}

#undef CASE1
#undef CASE2

} // namespace python
} // namespace fern

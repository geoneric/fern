// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#include "fern/python/algorithm/core/if.h"
#include "fern/core/data_customization_point/scalar.h"
#include "fern/python/feature/detail/data_customization_point/masked_raster.h"
#include "fern/algorithm/policy/policies.h"
#include "fern/algorithm/core/if.h"
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
    T2 const& true_value,
    detail::MaskedRaster<T3> const& false_value,
    detail::MaskedRaster<R>& result)
{
    fa::InputNoDataPolicies<
        fa::DetectNoData<detail::MaskedRaster<T1>>,
        fa::SkipNoData,
        fa::DetectNoData<detail::MaskedRaster<T3>>
    > input_no_data_policy{
        fa::DetectNoData<detail::MaskedRaster<T1>>{condition},
        fa::SkipNoData{},
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
    T2 const& true_value,
    detail::MaskedRaster<T3> const& false_value)
{
    using R = T3;
    auto handle = std::make_shared<detail::MaskedRaster<R>>(condition.sizes(),
        condition.origin(), condition.cell_sizes());
    if_(execution_policy, condition, true_value, false_value, *handle);
    return std::make_shared<MaskedRaster>(handle);
}

} // Anonymous namespace


#define CASE2(                                 \
    value_type_enum3,                          \
    value_type3,                               \
    value_type1)                               \
case value_type_enum3: {                       \
    result = if_(                              \
        execution_policy,                      \
        condition->raster<value_type1>(),      \
        static_cast<value_type3>(true_value),  \
        false_value->raster<value_type3>());   \
    break;                                     \
}

#define CASE1(              \
    value_type_enum1,       \
    value_type1,            \
    value_type_enum3)       \
case value_type_enum1: {    \
    SWITCH_ON_VALUE_TYPE2(  \
        value_type_enum3,   \
        CASE2,              \
        value_type1)        \
    break;                  \
}

MaskedRasterHandle if_(
    fa::ExecutionPolicy& execution_policy,
    MaskedRasterHandle const& condition,
    int64_t true_value,
    MaskedRasterHandle const& false_value)
{
    MaskedRasterHandle result;
    SWITCH_ON_VALUE_TYPE1(condition->value_type(), CASE1,
        false_value->value_type())
    return result;
}


MaskedRasterHandle if_(
    fa::ExecutionPolicy& execution_policy,
    MaskedRasterHandle const& condition,
    double true_value,
    MaskedRasterHandle const& false_value)
{
    MaskedRasterHandle result;
    SWITCH_ON_VALUE_TYPE1(condition->value_type(), CASE1,
        false_value->value_type())
    return result;
}

#undef CASE1
#undef CASE2

} // namespace python
} // namespace fern

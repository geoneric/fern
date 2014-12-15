#include "fern/python_extension/algorithm/core/if.h"
#include "fern/python_extension/algorithm/core/unite_no_data.h"
#include "fern/algorithm/core/if.h"
#include "fern/python_extension/core/switch_on_value_type.h"


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
    fern::MaskedRaster<T1, 2> const& condition,
    T2 const& true_value,
    fern::MaskedRaster<T3, 2> const& false_value,
    fern::MaskedRaster<R, 2>& result)
{
    using InputNoDataPolicy = fa::DetectNoDataByValue<fern::Mask<2>>;
    using OutputNoDataPolicy = fa::MarkNoDataByValue<fern::Mask<2>>;

    InputNoDataPolicy input_no_data_policy(result.mask(), true);
    OutputNoDataPolicy output_no_data_policy(result.mask(), true);

    fa::core::if_(input_no_data_policy, output_no_data_policy,
        execution_policy, condition, true_value, false_value, result);
}


template<
    typename T1,
    typename T2,
    typename T3>
MaskedRasterHandle if_(
    fa::ExecutionPolicy& execution_policy,
    fern::MaskedRaster<T1, 2> const& condition,
    T2 const& true_value,
    fern::MaskedRaster<T3, 2> const& false_value)
{
    auto sizes = extents[condition.shape()[0]][condition.shape()[1]];
    using R = T3;
    auto handle = std::make_shared<fern::MaskedRaster<R, 2>>(sizes,
        condition.transformation());
    unite_no_data(execution_policy, condition, false_value, *handle);
    if_(execution_policy, condition, true_value, false_value, *handle);
    return std::make_shared<MaskedRaster>(handle);
}

} // Anonymous namespace


#define CASE2(                                \
    value_type_enum3,                         \
    value_type3,                              \
    value_type1)                              \
case value_type_enum3: {                      \
    result = if_(                             \
        execution_policy,                     \
        condition->raster<value_type1>(),     \
        true_value,                           \
        false_value->raster<value_type3>());  \
    break;                                    \
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

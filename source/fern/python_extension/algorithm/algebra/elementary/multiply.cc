#include "fern/python_extension/algorithm/algebra/elementary/multiply.h"
#include "fern/python_extension/feature/detail/masked_raster_traits.h"
#include "fern/algorithm/policy/policies.h"
#include "fern/algorithm/algebra/elementary/multiply.h"
#include "fern/python_extension/core/switch_on_value_type.h"


namespace fa = fern::algorithm;


namespace fern {
namespace python {
namespace {

template<
    typename T1,
    typename T2,
    typename R>
void multiply(
    fa::ExecutionPolicy& execution_policy,
    detail::MaskedRaster<T1> const& lhs,
    detail::MaskedRaster<T2> const& rhs,
    detail::MaskedRaster<R>& result)
{
    fa::InputNoDataPolicies<
        fa::DetectNoData<detail::MaskedRaster<T1>>,
        fa::DetectNoData<detail::MaskedRaster<T2>>
    > input_no_data_policy{
        fa::DetectNoData<detail::MaskedRaster<T1>>{lhs},
        fa::DetectNoData<detail::MaskedRaster<T2>>{rhs}};
    fa::MarkNoData<detail::MaskedRaster<R>> output_no_data_policy(result);

    fa::algebra::multiply<fa::multiply::OutOfRangePolicy>(input_no_data_policy,
        output_no_data_policy, execution_policy, lhs, rhs, result);
}


template<
    typename T1,
    typename T2,
    typename R>
void multiply(
    fa::ExecutionPolicy& execution_policy,
    detail::MaskedRaster<T1> const& lhs,
    T2 const& rhs,
    detail::MaskedRaster<R>& result)
{
    fa::InputNoDataPolicies<
        fa::DetectNoData<detail::MaskedRaster<T1>>,
        fa::SkipNoData<>
    > input_no_data_policy{
        fa::DetectNoData<detail::MaskedRaster<T1>>{lhs},
        fa::SkipNoData<>{}};
    fa::MarkNoData<detail::MaskedRaster<R>> output_no_data_policy(result);

    fa::algebra::multiply<fa::multiply::OutOfRangePolicy>(input_no_data_policy,
        output_no_data_policy, execution_policy, lhs, rhs, result);
}


template<
    typename T1,
    typename T2,
    typename R>
void multiply(
    fa::ExecutionPolicy& execution_policy,
    T1 const& lhs,
    detail::MaskedRaster<T2> const& rhs,
    detail::MaskedRaster<R>& result)
{
    fa::InputNoDataPolicies<
        fa::SkipNoData<>,
        fa::DetectNoData<detail::MaskedRaster<T2>>
    > input_no_data_policy{
        fa::SkipNoData<>{},
        fa::DetectNoData<detail::MaskedRaster<T2>>{rhs}};
    fa::MarkNoData<detail::MaskedRaster<R>> output_no_data_policy(result);

    fa::algebra::multiply<fa::multiply::OutOfRangePolicy>(input_no_data_policy,
        output_no_data_policy, execution_policy, lhs, rhs, result);
}


template<
    typename T1,
    typename T2>
void imultiply(
    fa::ExecutionPolicy& execution_policy,
    detail::MaskedRaster<T1>& self,
    detail::MaskedRaster<T2> const& other)
{
    // TODO Assert raster properties.
    multiply(execution_policy, self, other, self);
}


template<
    typename T1,
    typename T2>
MaskedRasterHandle multiply(
    fa::ExecutionPolicy& execution_policy,
    detail::MaskedRaster<T1> const& lhs,
    detail::MaskedRaster<T2> const& rhs)
{
    // TODO Assert raster properties.
    using R = algorithm::multiply::result_type<T1, T2>;
    auto handle = std::make_shared<detail::MaskedRaster<R>>(lhs.sizes(),
        lhs.origin(), lhs.cell_sizes());
    multiply(execution_policy, lhs, rhs, *handle);
    return std::make_shared<MaskedRaster>(handle);
}


template<
    typename T1,
    typename T2>
MaskedRasterHandle multiply(
    fa::ExecutionPolicy& execution_policy,
    detail::MaskedRaster<T1> const& lhs,
    T2 const& rhs)
{
    using R = T1; // algorithm::multiply::result_type<T1, T2>;
    auto handle = std::make_shared<detail::MaskedRaster<R>>(lhs.sizes(),
        lhs.origin(), lhs.cell_sizes());
    multiply(execution_policy, lhs, rhs, *handle);
    return std::make_shared<MaskedRaster>(handle);
}


template<
    typename T1,
    typename T2>
MaskedRasterHandle multiply(
    fa::ExecutionPolicy& execution_policy,
    T1 const& lhs,
    detail::MaskedRaster<T2> const& rhs)
{
    using R = T2; // algorithm::multiply::result_type<T1, T2>;
    auto handle = std::make_shared<detail::MaskedRaster<R>>(rhs.sizes(),
        rhs.origin(), rhs.cell_sizes());
    multiply(execution_policy, lhs, rhs, *handle);
    return std::make_shared<MaskedRaster>(handle);
}

} // Anonymous namespace


#define CASE2(                          \
    value_type_enum2,                   \
    value_type2,                        \
    value_type1)                        \
case value_type_enum2: {                \
    imultiply(                          \
        execution_policy,               \
        self->raster<value_type1>(),    \
        other->raster<value_type2>());  \
    break;                              \
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

MaskedRasterHandle& imultiply(
    fa::ExecutionPolicy& execution_policy,
    MaskedRasterHandle& self,
    MaskedRasterHandle const& other)
{
    SWITCH_ON_VALUE_TYPE1(self->value_type(), CASE1, other->value_type());
    return self;
}

#undef CASE1
#undef CASE2


#define CASE2(                            \
    value_type_enum2,                     \
    value_type2,                          \
    value_type1)                          \
case value_type_enum2: {                  \
    result = multiply(                    \
        execution_policy,                 \
        raster1->raster<value_type1>(),   \
        raster2->raster<value_type2>());  \
    break;                                \
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

MaskedRasterHandle multiply(
    fa::ExecutionPolicy& execution_policy,
    MaskedRasterHandle const& raster1,
    MaskedRasterHandle const& raster2)
{
    MaskedRasterHandle result;
    SWITCH_ON_VALUE_TYPE1(raster1->value_type(), CASE1, raster2->value_type());
    return result;
}

#undef CASE1
#undef CASE2


#define CASE(                            \
    value_type_enum2,                    \
    value_type2)                         \
case value_type_enum2: {                 \
    result = multiply(                   \
        execution_policy,               \
        value,                           \
        raster->raster<value_type2>());  \
    break;                               \
}

MaskedRasterHandle multiply(
    fa::ExecutionPolicy& execution_policy,
    int64_t value,
    MaskedRasterHandle const& raster)
{
    MaskedRasterHandle result;
    SWITCH_ON_VALUE_TYPE(raster->value_type(), CASE)
    return result;
}


MaskedRasterHandle multiply(
    fa::ExecutionPolicy& execution_policy,
    double value,
    MaskedRasterHandle const& raster)
{
    MaskedRasterHandle result;
    SWITCH_ON_VALUE_TYPE(raster->value_type(), CASE)
    return result;
}

#undef CASE


#define CASE(                           \
    value_type_enum1,                   \
    value_type1)                        \
case value_type_enum1: {                \
    result = multiply(                  \
        execution_policy,               \
        raster->raster<value_type1>(),  \
        value);                         \
    break;                              \
}

MaskedRasterHandle multiply(
    fa::ExecutionPolicy& execution_policy,
    MaskedRasterHandle const& raster,
    int64_t value)
{
    MaskedRasterHandle result;
    SWITCH_ON_VALUE_TYPE(raster->value_type(), CASE)
    return result;
}


MaskedRasterHandle multiply(
    fa::ExecutionPolicy& execution_policy,
    MaskedRasterHandle const& raster,
    double value)
{
    MaskedRasterHandle result;
    SWITCH_ON_VALUE_TYPE(raster->value_type(), CASE)
    return result;
}

#undef CASE

} // namespace python
} // namespace fern

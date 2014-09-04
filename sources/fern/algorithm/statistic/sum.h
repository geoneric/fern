#pragma once
#include "fern/algorithm/algebra/elementary/add.h"
#include "fern/algorithm/statistic/detail/sum.h"


namespace fern {
namespace sum {

template<
    class Value1,
    class Value2,
    class Result>
using OutOfRangePolicy = add::OutOfRangePolicy<Value1, Value2, Result>;

} // namespace sum


namespace statistic {

//! Calculate the sum of \a value and write the result to \a result.
/*!
    \ingroup       statistic
    \sa            fern::unary_aggregate_operation

    The value type of \a value must be arithmetic and not `bool`. The value
    type of \a result must be equal to the value type of \a value.
*/
template<
    template<class, class, class> class OutOfRangePolicy,
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class ExecutionPolicy,
    class Value,
    class Result
>
void sum(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    ExecutionPolicy const& execution_policy,
    Value const& value,
    Result& result)
{
    FERN_STATIC_ASSERT(std::is_arithmetic, value_type<Value>)
    FERN_STATIC_ASSERT(!std::is_same, value_type<Value>, bool)
    FERN_STATIC_ASSERT(std::is_same, value_type<Result>, value_type<Value>)

    sum::detail::sum<OutOfRangePolicy>(input_no_data_policy,
        output_no_data_policy, execution_policy, value, result);
}


/*!
    \ingroup       statistic
    \overload
*/
template<
    template<class, class, class> class OutOfRangePolicy,
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class ExecutionPolicy,
    class Value,
    class Result
>
void sum(
    ExecutionPolicy const& execution_policy,
    Value const& value,
    Result& result)
{
    OutputNoDataPolicy output_no_data_policy;
    sum<OutOfRangePolicy>(InputNoDataPolicy(), output_no_data_policy,
        execution_policy, value, result);
}


/*!
    \ingroup       statistic
    \overload
*/
template<
    class ExecutionPolicy,
    class Value,
    class Result
>
void sum(
    ExecutionPolicy const& execution_policy,
    Value const& value,
    Result& result)
{
    using InputNoDataPolicy = SkipNoData;
    using OutputNoDataPolicy = DontMarkNoData;

    OutputNoDataPolicy output_no_data_policy;
    sum<binary::DiscardRangeErrors>(InputNoDataPolicy(),
        output_no_data_policy, execution_policy, value, result);
}

} // namespace statistic
} // namespace fern

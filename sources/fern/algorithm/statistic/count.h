#pragma once
#include "fern/algorithm/statistic/detail/count.h"


namespace fern {
namespace algorithm {
namespace statistic {

//! Determine how many occurrences of \a value2 are in \a value1 and write the result to \a result.
/*!
    \ingroup       statistic
    \sa            fern::binary_aggregate_operation,
                   @ref fern_algorithm_statistics

    The value type of \a value must be arithmetic. The value
    type of \a result must be integral.
*/
template<
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class ExecutionPolicy,
    class Value,
    class Result
>
void count(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    ExecutionPolicy const& execution_policy,
    Value const& values,
    value_type<Value> const& value,
    Result& result)
{
    FERN_STATIC_ASSERT(std::is_arithmetic, value_type<Value>)
    FERN_STATIC_ASSERT(std::is_integral, value_type<Result>)

    count::detail::count<>(input_no_data_policy, output_no_data_policy,
        execution_policy, values, value, result);
}


/*!
    \ingroup       statistic
    \overload
*/
template<
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class ExecutionPolicy,
    class Value,
    class Result
>
void count(
    ExecutionPolicy const& execution_policy,
    Value const& values,
    value_type<Value> const& value,
    Result& result)
{
    OutputNoDataPolicy output_no_data_policy;
    count<>(InputNoDataPolicy(), output_no_data_policy, execution_policy,
        values, value, result);
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
void count(
    ExecutionPolicy const& execution_policy,
    Value const& values,
    value_type<Value> const& value,
    Result& result)
{
    using InputNoDataPolicy = SkipNoData<>;
    using OutputNoDataPolicy = DontMarkNoData;

    OutputNoDataPolicy output_no_data_policy;
    count<>(InputNoDataPolicy(), output_no_data_policy, execution_policy,
        values, value, result);
}

} // namespace statistic
} // namespace algorithm
} // namespace fern

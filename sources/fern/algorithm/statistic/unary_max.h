#pragma once
#include "fern/algorithm/policy/policies.h"
#include "fern/algorithm/statistic/detail/unary_max.h"


namespace fern {
namespace algorithm {
namespace statistic {

//! Determine the maximum value of \a value and write the result to \a result.
/*!
    \ingroup       statistic
    \sa            fern::unary_aggregate_operation,
                   @ref fern_algorithm_statistics

    The value type of \a value must be arithmetic and not `bool`. The value
    type of \a result must be equal to the value type of \a value.
*/
template<
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class ExecutionPolicy,
    class Value,
    class Result
>
void unary_max(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    ExecutionPolicy const& execution_policy,
    Value const& value,
    Result& result)
{
    FERN_STATIC_ASSERT(std::is_arithmetic, value_type<Value>)
    FERN_STATIC_ASSERT(!std::is_same, value_type<Value>, bool)
    FERN_STATIC_ASSERT(std::is_same, value_type<Result>, value_type<Value>)

    unary_max::detail::unary_max<>(input_no_data_policy, output_no_data_policy,
        execution_policy, value, result);
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
void unary_max(
    ExecutionPolicy const& execution_policy,
    Value const& value,
    Result& result)
{
    OutputNoDataPolicy output_no_data_policy;
    unary_max<>(InputNoDataPolicy(), output_no_data_policy, execution_policy,
        value, result);
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
void unary_max(
    ExecutionPolicy const& execution_policy,
    Value const& value,
    Result& result)
{
    using InputNoDataPolicy = SkipNoData<>;
    using OutputNoDataPolicy = DontMarkNoData;

    OutputNoDataPolicy output_no_data_policy;
    unary_max<>(InputNoDataPolicy(), output_no_data_policy, execution_policy,
        value, result);
}

} // namespace statistic
} // namespace algorithm
} // namespace fern

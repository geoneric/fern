#pragma once
#include "fern/core/assert.h"
#include "fern/algorithm/policy/policies.h"
#include "fern/algorithm/core/detail/fill.h"


namespace fern {
namespace core {

//! Fill \a result with \a value.
/*!
    \sa            fern::unary_disaggregate_operation

    The value type of \a Result must be equal to \a Value.
*/
template<
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class ExecutionPolicy,
    class Value,
    class Result>
void fill(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    ExecutionPolicy const& execution_policy,
    Value const& value,
    Result& result)
{
    FERN_STATIC_ASSERT(std::is_same, value_type<Result>, Value)

    fill::detail::fill<>(input_no_data_policy,
        output_no_data_policy, execution_policy, value, result);
}


/*!
  \overload
*/
template<
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class ExecutionPolicy,
    class Value,
    class Result>
void fill(
    ExecutionPolicy const& execution_policy,
    Value const& value,
    Result& result)
{
    OutputNoDataPolicy output_no_data_policy;
    fill<>(InputNoDataPolicy(), output_no_data_policy,
        execution_policy, value, result);
}


/*!
  \overload
*/
template<
    class ExecutionPolicy,
    class Value,
    class Result>
void fill(
    ExecutionPolicy const& execution_policy,
    Value const& value,
    Result& result)
{
    using InputNoDataPolicy = SkipNoData;
    using OutputNoDataPolicy = DontMarkNoData;

    OutputNoDataPolicy output_no_data_policy;
    fill<>(InputNoDataPolicy(), output_no_data_policy, execution_policy,
        value, result);
}

} // namespace core
} // namespace fern

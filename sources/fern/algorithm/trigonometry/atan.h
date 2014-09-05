#pragma once
#include "fern/core/assert.h"
#include "fern/algorithm/policy/policies.h"
#include "fern/algorithm/trigonometry/detail/atan.h"


namespace fern {
namespace trigonometry {

//! Calculate the arc tangent of \a value and write the result to \a result.
/*!
    \ingroup       trigonometry
    \sa            fern::unary_local_operation

    The value types of \a value and \a result must be floating point and the
    same.
*/
template<
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class ExecutionPolicy,
    class Value,
    class Result>
void atan(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    ExecutionPolicy const& execution_policy,
    Value const& value,
    Result& result)
{
    FERN_STATIC_ASSERT(std::is_floating_point, value_type<Value>)
    FERN_STATIC_ASSERT(std::is_same, value_type<Result>, value_type<Value>)

    atan::detail::atan<>(input_no_data_policy,
        output_no_data_policy, execution_policy, value, result);
}


/*!
    \ingroup       trigonometry
    \overload
*/
template<
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class ExecutionPolicy,
    class Value,
    class Result>
void atan(
    ExecutionPolicy const& execution_policy,
    Value const& value,
    Result& result)
{
    OutputNoDataPolicy output_no_data_policy;
    atan<>(InputNoDataPolicy(), output_no_data_policy, execution_policy,
        value, result);
}


/*!
    \ingroup       trigonometry
    \overload
*/
template<
    class ExecutionPolicy,
    class Value,
    class Result>
void atan(
    ExecutionPolicy const& execution_policy,
    Value const& value,
    Result& result)
{
    using InputNoDataPolicy = SkipNoData;
    using OutputNoDataPolicy = DontMarkNoData;

    OutputNoDataPolicy output_no_data_policy;
    atan<>(InputNoDataPolicy(), output_no_data_policy, execution_policy,
        value, result);
}

} // namespace trigonometry
} // namespace fern

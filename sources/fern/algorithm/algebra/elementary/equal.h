#pragma once
#include "fern/core/assert.h"
#include "fern/algorithm/policy/policies.h"
#include "fern/algorithm/algebra/elementary/detail/equal.h"


namespace fern {
namespace algorithm {
namespace algebra {

//! Determine whether \a value1 is equal to \a value2 and write the result to \a result.
/*!
    \ingroup       elementary
    \sa            fern::binary_local_operation,
                   @ref fern_algorithm_algebra_elementary

    The value types of \a value1 and \a value2 must be arithmetic.
    The value type of \a result must be arithmetic.
*/
template<
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class ExecutionPolicy,
    class Value1,
    class Value2,
    class Result>
void equal(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    ExecutionPolicy const& execution_policy,
    Value1 const& value1,
    Value2 const& value2,
    Result& result)
{
    FERN_STATIC_ASSERT(std::is_arithmetic, value_type<Value1>)
    FERN_STATIC_ASSERT(std::is_arithmetic, value_type<Value2>)
    FERN_STATIC_ASSERT(std::is_arithmetic, value_type<Result>)

    equal::detail::equal<>(input_no_data_policy, output_no_data_policy,
        execution_policy, value1, value2, result);
}


/*!
    \ingroup       elementary
    \overload
*/
template<
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class ExecutionPolicy,
    class Value1,
    class Value2,
    class Result>
void equal(
    ExecutionPolicy const& execution_policy,
    Value1 const& value1,
    Value2 const& value2,
    Result& result)
{
    OutputNoDataPolicy output_no_data_policy;
    equal<>(InputNoDataPolicy(),
        output_no_data_policy, execution_policy, value1, value2, result);
}


/*!
    \ingroup       elementary
    \overload
*/
template<
    class ExecutionPolicy,
    class Value1,
    class Value2,
    class Result>
void equal(
    ExecutionPolicy const& execution_policy,
    Value1 const& value1,
    Value2 const& value2,
    Result& result)
{
    using InputNoDataPolicy = SkipNoData<>;
    using OutputNoDataPolicy = DontMarkNoData;

    OutputNoDataPolicy output_no_data_policy;
    equal<>(InputNoDataPolicy(), output_no_data_policy, execution_policy,
        value1, value2, result);
}

} // namespace algebra
} // namespace algorithm
} // namespace fern

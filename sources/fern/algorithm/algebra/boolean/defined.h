#pragma once
#include "fern/core/assert.h"
#include "fern/algorithm/policy/policies.h"
#include "fern/algorithm/algebra/boolean/detail/defined.h"


namespace fern {
namespace algebra {

//! Determing which values are defined and write the result to \a result.
/*!
    \sa            fern::nullary_local_operation.
    \warning       For each location in \a result, the \a InputNoDataPolicy
                   must be able to tell whether the input value is no-data
                   or not.

    Whether or not a value is defind depends on the input-not-data policy.
    Therefore, this algorithm doesn't need a value to be passed in.

    The value type of \a result must be integral.
*/
template<
    class InputNoDataPolicy,
    class ExecutionPolicy,
    class Result
>
void defined(
    InputNoDataPolicy const& input_no_data_policy,
    ExecutionPolicy const& execution_policy,
    Result& result)
{
    FERN_STATIC_ASSERT(std::is_integral, value_type<Result>)

    defined::detail::defined<>(input_no_data_policy, execution_policy, result);
}


/*!
  \overload
*/
template<
    class InputNoDataPolicy,
    class ExecutionPolicy,
    class Result
>
void defined(
    ExecutionPolicy const& execution_policy,
    Result& result)
{
    defined<>(InputNoDataPolicy(), execution_policy, result);
}


/*!
  \overload
*/
template<
    class ExecutionPolicy,
    class Result
>
void defined(
    ExecutionPolicy const& execution_policy,
    Result& result)
{
    using InputNoDataPolicy = SkipNoData;

    defined<>(InputNoDataPolicy(), execution_policy, result);
}

} // namespace algebra
} // namespace fern

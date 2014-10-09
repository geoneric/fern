#pragma once
#include "fern/core/assert.h"
#include "fern/algorithm/policy/policies.h"
#include "fern/algorithm/algebra/boole/detail/defined.h"


namespace fern {
namespace algorithm {
namespace algebra {

/*!
    @ingroup    fern_algorithm_algebra_boole_group
    @brief      Determing which values are defined and write the result to
                @a result.
    @warning    For each location in @a result, the @a InputNoDataPolicy
                must be able to tell whether the input value is no-data
                or not.
    @sa         fern::algorithm::nullary_local_operation

    Whether or not a value is defind depends on the input-not-data policy.
    Therefore, this algorithm doesn't need a value to be passed in.

    The value type of @a result must be arithmetic.
*/
template<
    typename InputNoDataPolicy,
    typename ExecutionPolicy,
    typename Result
>
void defined(
    InputNoDataPolicy const& input_no_data_policy,
    ExecutionPolicy const& execution_policy,
    Result& result)
{
    FERN_STATIC_ASSERT(std::is_arithmetic, value_type<Result>)

    defined::detail::defined<>(input_no_data_policy, execution_policy, result);
}


/*!
    @ingroup    fern_algorithm_algebra_boole_group
    @overload
*/
template<
    typename InputNoDataPolicy,
    typename ExecutionPolicy,
    typename Result
>
void defined(
    ExecutionPolicy const& execution_policy,
    Result& result)
{
    defined<>(InputNoDataPolicy(), execution_policy, result);
}


/*!
    @ingroup    fern_algorithm_algebra_boole_group
    @overload
*/
template<
    typename ExecutionPolicy,
    typename Result
>
void defined(
    ExecutionPolicy const& execution_policy,
    Result& result)
{
    using InputNoDataPolicy = SkipNoData<>;

    defined<>(InputNoDataPolicy(), execution_policy, result);
}

} // namespace algebra
} // namespace algorithm
} // namespace fern

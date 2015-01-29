#pragma once
#include "fern/core/assert.h"
#include "fern/algorithm/policy/policies.h"
#include "fern/algorithm/algebra/elementary/detail/less_equal.h"


namespace fern {
namespace algorithm {
namespace algebra {

/*!
    @ingroup    fern_algorithm_algebra_elementary_group
    @brief      Determine whether @a value1 is less than or equal to
                @a value2 and write the result to @a result.
    @sa         fern::algorithm::binary_local_operation

    The value types of @a value1 and @a value2 must be arithmetic and not
    `bool`. The value type of @a result must be arithmetic.
*/
template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename ExecutionPolicy,
    typename Value1,
    typename Value2,
    typename Result>
void less_equal(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    ExecutionPolicy& execution_policy,
    Value1 const& value1,
    Value2 const& value2,
    Result& result)
{
    FERN_STATIC_ASSERT(std::is_arithmetic, value_type<Value1>)
    FERN_STATIC_ASSERT(!std::is_same, value_type<Value1>, bool)
    FERN_STATIC_ASSERT(std::is_arithmetic, value_type<Value2>)
    FERN_STATIC_ASSERT(!std::is_same, value_type<Value2>, bool)
    FERN_STATIC_ASSERT(std::is_arithmetic, value_type<Result>)

    less_equal::detail::less_equal<>(input_no_data_policy,
        output_no_data_policy, execution_policy, value1, value2, result);
}


/*!
    @ingroup       fern_algorithm_algebra_elementary_group
    @overload
*/
template<
    typename ExecutionPolicy,
    typename Value1,
    typename Value2,
    typename Result>
void less_equal(
    ExecutionPolicy& execution_policy,
    Value1 const& value1,
    Value2 const& value2,
    Result& result)
{
    using InputNoDataPolicy = InputNoDataPolicies<SkipNoData, SkipNoData>;
    using OutputNoDataPolicy = DontMarkNoData;

    OutputNoDataPolicy output_no_data_policy;
    less_equal<>(InputNoDataPolicy{{}, {}}, output_no_data_policy,
        execution_policy, value1, value2, result);
}

} // namespace algebra
} // namespace algorithm
} // namespace fern

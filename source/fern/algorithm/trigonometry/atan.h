#pragma once
#include "fern/core/assert.h"
#include "fern/algorithm/policy/policies.h"
#include "fern/algorithm/trigonometry/detail/atan.h"


namespace fern {
namespace algorithm {
namespace trigonometry {

/*!
    @ingroup    fern_algorithm_trigonometry_group
    @brief      Calculate the arc tangent of @a value and write the
                result to @a result.
    @sa         fern::algorithm::unary_local_operation

    The value types of @a value and @a result must be floating point and the
    same.
*/
template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename ExecutionPolicy,
    typename Value,
    typename Result>
void atan(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    ExecutionPolicy& execution_policy,
    Value const& value,
    Result& result)
{
    FERN_STATIC_ASSERT(std::is_floating_point, value_type<Value>)
    FERN_STATIC_ASSERT(std::is_same, value_type<Result>, value_type<Value>)

    atan::detail::atan<>(input_no_data_policy,
        output_no_data_policy, execution_policy, value, result);
}


/*!
    @ingroup    fern_algorithm_trigonometry_group
    @overload
*/
template<
    typename ExecutionPolicy,
    typename Value,
    typename Result>
void atan(
    ExecutionPolicy& execution_policy,
    Value const& value,
    Result& result)
{
    using InputNoDataPolicy = InputNoDataPolicies<SkipNoData>;
    using OutputNoDataPolicy = DontMarkNoData;

    OutputNoDataPolicy output_no_data_policy;
    atan<>(InputNoDataPolicy{{}}, output_no_data_policy, execution_policy,
        value, result);
}

} // namespace trigonometry
} // namespace algorithm
} // namespace fern

#pragma once
#include "fern/algorithm/policy/policies.h"
#include "fern/algorithm/statistic/detail/unary_min.h"


namespace fern {
namespace algorithm {
namespace statistic {

/*!
    @ingroup    fern_algorithm_statistic_group
    @brief      Determine the minimum value of @a value and write the
                result to @a result.
    @sa         fern::algorithm::unary_aggregate_operation

    The value type of @a value must be arithmetic and not `bool`. The value
    type of @a result must be equal to the value type of @a value.
*/
template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename ExecutionPolicy,
    typename Value,
    typename Result
>
void unary_min(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    ExecutionPolicy const& execution_policy,
    Value const& value,
    Result& result)
{
    FERN_STATIC_ASSERT(std::is_arithmetic, value_type<Value>)
    FERN_STATIC_ASSERT(!std::is_same, value_type<Value>, bool)
    FERN_STATIC_ASSERT(std::is_same, value_type<Result>, value_type<Value>)

    unary_min::detail::unary_min<>(input_no_data_policy, output_no_data_policy,
        execution_policy, value, result);
}


/*!
    @ingroup    fern_algorithm_statistic_group
    @overload
*/
template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename ExecutionPolicy,
    typename Value,
    typename Result
>
void unary_min(
    ExecutionPolicy const& execution_policy,
    Value const& value,
    Result& result)
{
    OutputNoDataPolicy output_no_data_policy;
    unary_min<>(InputNoDataPolicy(), output_no_data_policy, execution_policy,
        value, result);
}


/*!
    @ingroup    fern_algorithm_statistic_group
    @overload
*/
template<
    typename ExecutionPolicy,
    typename Value,
    typename Result
>
void unary_min(
    ExecutionPolicy const& execution_policy,
    Value const& value,
    Result& result)
{
    using InputNoDataPolicy = SkipNoData<>;
    using OutputNoDataPolicy = DontMarkNoData;

    OutputNoDataPolicy output_no_data_policy;
    unary_min<>(InputNoDataPolicy(), output_no_data_policy, execution_policy,
        value, result);
}

} // namespace statistic
} // namespace algorithm
} // namespace fern

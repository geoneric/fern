#pragma once
#include "fern/algorithm/algebra/elementary/add.h"
#include "fern/algorithm/statistic/detail/sum.h"


namespace fern {
namespace algorithm {
namespace sum {

/*!
    @ingroup    fern_algorithm_statistic_group
*/
template<
    typename Value1,
    typename Value2,
    typename Result>
using OutOfRangePolicy = add::OutOfRangePolicy<Value1, Value2, Result>;

} // namespace sum


namespace statistic {

/*!
    @ingroup    fern_algorithm_statistic_group
    @brief      Calculate the sum of @a value and write the result to
                @a result.
    @sa         fern::algorithm::unary_aggregate_operation,
                sum::OutOfRangePolicy

    The value type of @a value must be arithmetic and not `bool`. The value
    type of @a result must be equal to the value type of @a value.
*/
template<
    template<typename, typename, typename> class OutOfRangePolicy,
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename ExecutionPolicy,
    typename Value,
    typename Result
>
void sum(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    ExecutionPolicy const& execution_policy,
    Value const& value,
    Result& result)
{
    FERN_STATIC_ASSERT(std::is_arithmetic, value_type<Value>)
    FERN_STATIC_ASSERT(!std::is_same, value_type<Value>, bool)
    FERN_STATIC_ASSERT(std::is_same, value_type<Result>, value_type<Value>)

    sum::detail::sum<OutOfRangePolicy>(input_no_data_policy,
        output_no_data_policy, execution_policy, value, result);
}


/*!
    @ingroup    fern_algorithm_statistic_group
    @overload
*/
template<
    template<typename, typename, typename> class OutOfRangePolicy,
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename ExecutionPolicy,
    typename Value,
    typename Result
>
void sum(
    ExecutionPolicy const& execution_policy,
    Value const& value,
    Result& result)
{
    OutputNoDataPolicy output_no_data_policy;
    sum<OutOfRangePolicy>(InputNoDataPolicy(), output_no_data_policy,
        execution_policy, value, result);
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
void sum(
    ExecutionPolicy const& execution_policy,
    Value const& value,
    Result& result)
{
    using InputNoDataPolicy = SkipNoData<>;
    using OutputNoDataPolicy = DontMarkNoData;

    OutputNoDataPolicy output_no_data_policy;
    sum<binary::DiscardRangeErrors>(InputNoDataPolicy(),
        output_no_data_policy, execution_policy, value, result);
}

} // namespace statistic
} // namespace algorithm
} // namespace fern
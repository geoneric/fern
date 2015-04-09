// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
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
    ExecutionPolicy& execution_policy,
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
    typename ExecutionPolicy,
    typename Value,
    typename Result
>
void sum(
    ExecutionPolicy& execution_policy,
    Value const& value,
    Result& result)
{
    using InputNoDataPolicy = InputNoDataPolicies<SkipNoData>;
    using OutputNoDataPolicy = DontMarkNoData;

    OutputNoDataPolicy output_no_data_policy;
    sum<binary::DiscardRangeErrors>(InputNoDataPolicy{{}},
        output_no_data_policy, execution_policy, value, result);
}

} // namespace statistic
} // namespace algorithm
} // namespace fern

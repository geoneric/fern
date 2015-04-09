// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include "fern/algorithm/statistic/detail/count.h"


namespace fern {
namespace algorithm {
namespace statistic {

/*!
    @ingroup    fern_algorithm_statistic_group
    @brief      Determine how many occurrences of @a value2 are in @a
                value1 and write the result to @a result.
    @sa         fern::algorithm::binary_aggregate_operation

    The value type of @a value must be arithmetic. The value type of @a
    result must be integral.
*/
template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename ExecutionPolicy,
    typename Value,
    typename Result
>
void count(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    ExecutionPolicy& execution_policy,
    Value const& values,
    value_type<Value> const& value,
    Result& result)
{
    FERN_STATIC_ASSERT(std::is_arithmetic, value_type<Value>)
    FERN_STATIC_ASSERT(std::is_integral, value_type<Result>)

    count::detail::count<>(input_no_data_policy, output_no_data_policy,
        execution_policy, values, value, result);
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
void count(
    ExecutionPolicy& execution_policy,
    Value const& values,
    value_type<Value> const& value,
    Result& result)
{
    using InputNoDataPolicy = InputNoDataPolicies<SkipNoData, SkipNoData>;
    using OutputNoDataPolicy = DontMarkNoData;

    OutputNoDataPolicy output_no_data_policy;
    count<>(InputNoDataPolicy{{}, {}}, output_no_data_policy, execution_policy,
        values, value, result);
}

} // namespace statistic
} // namespace algorithm
} // namespace fern

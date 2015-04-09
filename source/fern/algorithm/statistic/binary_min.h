// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include "fern/algorithm/policy/policies.h"
#include "fern/algorithm/statistic/detail/binary_min.h"


namespace fern {
namespace algorithm {
namespace statistic {

/*!
    @ingroup    fern_algorithm_statistic_group
    @brief      Determine the elementwise minimum value of @a value1
                and @a value2 and write the result to @a result.
    @sa         fern::algorithm::binary_local_operation

    The value type of @a value1 and @a value2 must be arithmetic and not
    `bool`. They must also be the same. The value type of @a result must
    be equal to the value type of the values.
*/
template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename ExecutionPolicy,
    typename Value1,
    typename Value2,
    typename Result
>
void binary_min(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    ExecutionPolicy& execution_policy,
    Value1 const& value1,
    Value2 const& value2,
    Result& result)
{
    FERN_STATIC_ASSERT(std::is_arithmetic, value_type<Value1>)
    FERN_STATIC_ASSERT(std::is_arithmetic, value_type<Value2>)
    FERN_STATIC_ASSERT(!std::is_same, value_type<Value1>, bool)
    FERN_STATIC_ASSERT(std::is_same, value_type<Value1>, value_type<Value2>)
    FERN_STATIC_ASSERT(std::is_same, value_type<Result>, value_type<Value1>)

    binary_min::detail::binary_min<>(input_no_data_policy,
        output_no_data_policy, execution_policy, value1, value2, result);
}


/*!
    @ingroup    fern_algorithm_statistic_group
    @overload
*/
template<
    typename ExecutionPolicy,
    typename Value1,
    typename Value2,
    typename Result
>
void binary_min(
    ExecutionPolicy& execution_policy,
    Value1 const& value1,
    Value2 const& value2,
    Result& result)
{
    using InputNoDataPolicy = InputNoDataPolicies<SkipNoData, SkipNoData>;
    using OutputNoDataPolicy = DontMarkNoData;

    OutputNoDataPolicy output_no_data_policy;
    binary_min<>(InputNoDataPolicy{{}, {}}, output_no_data_policy,
        execution_policy, value1, value2, result);
}

} // namespace statistic
} // namespace algorithm
} // namespace fern

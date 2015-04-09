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
#include "fern/algorithm/core/detail/merge_no_data.h"


namespace fern {
namespace algorithm {
namespace core {

/*!
    @ingroup    fern_algorithm_core_group
    @brief      Merge the no-data of @a value with @a result.


    If an element from @a value contains a no-data, it will be marked as
    no-data in @a result too.

    - The value types of @a value and @a result are not relevant.
*/
template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename ExecutionPolicy,
    typename Value,
    typename Result>
void merge_no_data(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    ExecutionPolicy& execution_policy,
    Value const& value,
    Result& result)
{
    merge_no_data::detail::merge_no_data<>(input_no_data_policy,
        output_no_data_policy, execution_policy, value, result);
}


/*!
    @ingroup    fern_algorithm_core_group
    @overload
*/
template<
    typename ExecutionPolicy,
    typename Value,
    typename Result>
void merge_no_data(
    ExecutionPolicy& execution_policy,
    Value const& value,
    Result& result)
{
    using InputNoDataPolicy = InputNoDataPolicies<SkipNoData, SkipNoData>;
    using OutputNoDataPolicy = DontMarkNoData;

    OutputNoDataPolicy output_no_data_policy;
    merge_no_data<>(InputNoDataPolicy{{}, {}},
        output_no_data_policy, execution_policy, value, result);
}

} // namespace core
} // namespace algorithm
} // namespace fern

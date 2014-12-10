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
    ExecutionPolicy const& execution_policy,
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
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename ExecutionPolicy,
    typename Value,
    typename Result>
void merge_no_data(
    ExecutionPolicy const& execution_policy,
    Value const& value,
    Result& result)
{
    OutputNoDataPolicy output_no_data_policy;
    merge_no_data<>(InputNoDataPolicy(), output_no_data_policy,
        execution_policy, value, result);
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
    ExecutionPolicy const& execution_policy,
    Value const& value,
    Result& result)
{
    using InputNoDataPolicy = SkipNoData<SkipNoData<>, SkipNoData<>>;
    using OutputNoDataPolicy = DontMarkNoData;

    OutputNoDataPolicy output_no_data_policy;
    merge_no_data<>(InputNoDataPolicy(SkipNoData<>(), SkipNoData<>()),
        output_no_data_policy, execution_policy, value, result);
}

} // namespace core
} // namespace algorithm
} // namespace fern

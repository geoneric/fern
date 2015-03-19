#pragma once
#include "fern/core/data_traits.h"
#include "fern/core/assert.h"
#include "fern/algorithm/core/detail/clamp.h"
#include "fern/algorithm/policy/policies.h"


namespace fern {
namespace algorithm {
namespace core {

/*!
    @ingroup    fern_algorithm_core_group
    @brief      Limit the values in @a value to the range defined by
                @a lower_bound and @a upper_bound and write the resulting
                value to @a result.

    - The rank of @a Value must be larger or equal to the rank of
      @a LowerBound and @a UpperBound.
    - Value types of @a Value, @a UpperBound, @a LowerBound and @a Result
      must be arithmetic.
    - Value types of @a Value, @a UpperBound, @a LowerBound and @a Result
      must be the same.
    - The rank of @a Result must be equal to the rank of @a Value.
    - @a lower_bound and @a upper_bound must not contain no-data.
*/
template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename ExecutionPolicy,
    typename Value,
    typename LowerBound,
    typename UpperBound,
    typename Result>
void clamp(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    ExecutionPolicy& execution_policy,
    Value const& value,
    LowerBound const& lower_bound,
    UpperBound const& upper_bound,
    Result& result)
{

    static_assert(rank<Value>() == rank<Result>(), "");
    static_assert(rank<Value>() >= rank<LowerBound>(), "");
    static_assert(rank<Value>() >= rank<UpperBound>(), "");
    FERN_STATIC_ASSERT(std::is_same, value_type<UpperBound>, value_type<Value>)
    FERN_STATIC_ASSERT(std::is_same, value_type<LowerBound>, value_type<Value>)
    FERN_STATIC_ASSERT(std::is_same, value_type<Result>, value_type<Value>)

    clamp::detail::clamp<>(input_no_data_policy, output_no_data_policy,
        execution_policy, value, lower_bound, upper_bound, result);
}


/*!
    @ingroup    fern_algorithm_core_group
    @overload
*/
template<
    typename ExecutionPolicy,
    typename Value,
    typename LowerBound,
    typename UpperBound,
    typename Result>
void clamp(
    ExecutionPolicy& execution_policy,
    Value const& value,
    LowerBound const& lower_bound,
    UpperBound const& upper_bound,
    Result& result)
{
    using InputNoDataPolicy = InputNoDataPolicies<SkipNoData, SkipNoData,
        SkipNoData>;
    using OutputNoDataPolicy = DontMarkNoData;

    OutputNoDataPolicy output_no_data_policy;
    clamp<>(InputNoDataPolicy{{}, {}, {}},
        output_no_data_policy, execution_policy, value, lower_bound,
        upper_bound, result);
}

} // namespace core
} // namespace algorithm
} // namespace fern

#pragma once
#include "fern/core/assert.h"
#include "fern/core/base_class.h"
#include "fern/algorithm/policy/policies.h"
#include "fern/algorithm/core/detail/cast.h"


namespace fern {
namespace algorithm {
namespace cast {

/*!
    @ingroup    fern_algorithm_core_group
    @brief      Out-of-range policy for fern::cast algorithm.
    @sa         @ref fern_algorithm_policies_out_of_range_policy

    A source input value is considered out-of-range if the value cannot be
    represented by the target type.

    The value type of @a value and @a result must be arithmetic.
*/
template<
    typename Value,
    typename Result>
class OutOfRangePolicy
{

    FERN_STATIC_ASSERT(std::is_arithmetic, Value)
    FERN_STATIC_ASSERT(std::is_arithmetic, Result)

public:

    inline static bool within_range(
        Value const& value,
        Result const& result)
    {
        using value_tag =  base_class<number_category<Value>, integer_tag>;
        using result_tag = base_class<number_category<Result>, integer_tag>;

        return detail::dispatch::within_range<Value, Result, value_tag,
            result_tag>::calculate(value, result);
    }

};

} // namespace cast


namespace core {

/*!
    @ingroup    fern_algorithm_core_group
    @brief      Cast @a value and write the result to @a result.
    @sa         fern::algorithm::cast::OutOfRangePolicy,
                fern::algorithm::unary_local_operation

    The value type of @a value and @a result must be arithmetic.
*/
template<
    template<typename, typename> class OutOfRangePolicy,
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename ExecutionPolicy,
    typename Value,
    typename Result>
void cast(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    ExecutionPolicy& execution_policy,
    Value const& value,
    Result& result)
{
    FERN_STATIC_ASSERT(std::is_arithmetic, value_type<Value>)
    FERN_STATIC_ASSERT(std::is_arithmetic, value_type<Result>)

    cast::detail::cast<OutOfRangePolicy>(input_no_data_policy,
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
void cast(
    ExecutionPolicy& execution_policy,
    Value const& value,
    Result& result)
{
    using InputNoDataPolicy = InputNoDataPolicies<SkipNoData>;
    using OutputNoDataPolicy = DontMarkNoData;

    OutputNoDataPolicy output_no_data_policy;
    cast<unary::DiscardRangeErrors>(InputNoDataPolicy{{}},
        output_no_data_policy, execution_policy, value, result);
}

} // namespace core
} // namespace algorithm
} // namespace fern

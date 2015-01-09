#pragma once
#include "fern/core/assert.h"
#include "fern/core/base_class.h"
#include "fern/algorithm/policy/policies.h"
#include "fern/algorithm/algebra/elementary/detail/absolute.h"


namespace fern {
namespace algorithm {
namespace absolute {

/*!
    @ingroup    fern_algorithm_algebra_elementary_group
    @brief      Out of range policy for fern::algorithm::algebra::absolute
                algorithm.
    @sa         @ref fern_algorithm_policies_out_of_range_policy

    The logic for determining whether absolute's result is out of range depends
    on the types involved (unsigned integers, signed integers, floating
    points) and their sizes.

    - The value types of @a value1 and @a value2 must be arithmetic and not
      `bool`.
    - The value type of @a result must be equal to
      fern::algorithm::add::result_value_type<Value1, Value2>.
*/
template<
    typename Value,
    typename Result>
class OutOfRangePolicy
{

    FERN_STATIC_ASSERT(std::is_arithmetic, Value)
    FERN_STATIC_ASSERT(!std::is_same, value_type<Value>, bool)
    FERN_STATIC_ASSERT(std::is_arithmetic, Result)

public:

    inline static bool within_range(
        Value const& value,
        Result const& result)
    {
        using value_tag = base_class<number_category<Value>, integer_tag>;

        return detail::dispatch::within_range<Value, Result, value_tag>::
            calculate(value, result);
    }

};

} // namespace absolute


namespace algebra {

/*!
    @ingroup    fern_algorithm_algebra_elementary_group
    @brief      Determine the absolute value of @a value and write the
                result to @a result.
    @sa         fern::algorithm::absolute::OutOfRangePolicy,
                fern::algorithm::unary_local_operation

    The value type of @a value must be arithmetic and not `bool`. The value
    type of @a result must be equal to the value type of @a value.
*/
template<
    template<typename, typename> class OutOfRangePolicy,
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename ExecutionPolicy,
    typename Value,
    typename Result
>
void absolute(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    ExecutionPolicy const& execution_policy,
    Value const& value,
    Result& result)
{
    FERN_STATIC_ASSERT(std::is_arithmetic, value_type<Value>)
    FERN_STATIC_ASSERT(!std::is_same, value_type<Value>, bool)
    FERN_STATIC_ASSERT(std::is_arithmetic, value_type<Result>)

    absolute::detail::absolute<OutOfRangePolicy>(input_no_data_policy,
        output_no_data_policy, execution_policy, value, result);
}


/*!
    @ingroup    fern_algorithm_algebra_elementary_group
    @overload
*/
template<
    typename ExecutionPolicy,
    typename Value,
    typename Result
>
void absolute(
    ExecutionPolicy const& execution_policy,
    Value const& value,
    Result& result)
{
    using InputNoDataPolicy = InputNoDataPolicies<SkipNoData<>>;
    using OutputNoDataPolicy = DontMarkNoData;

    OutputNoDataPolicy output_no_data_policy;
    absolute<unary::DiscardRangeErrors>(InputNoDataPolicy{{}},
        output_no_data_policy, execution_policy, value, result);
}

} // namespace algebra
} // namespace algorithm
} // namespace fern

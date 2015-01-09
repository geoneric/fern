#pragma once
#include "fern/core/assert.h"
#include "fern/core/base_class.h"
#include "fern/algorithm/policy/policies.h"
#include "fern/algorithm/algebra/elementary/detail/subtract.h"


namespace fern {
namespace algorithm {
namespace subtract {

/*!
    @ingroup    fern_algorithm_algebra_elementary_group
    @brief      Determine the result type when subtracting instances of
                @a Value2 from @a Value1.
    @sa         fern::algorithm::subtract::result_value_type,
                fern::algorithm::Result
*/
template<
    typename Value1,
    typename Value2>
using result_type = typename fern::algorithm::Result<Value1, Value2>::type;


/*!
    @ingroup    fern_algorithm_algebra_elementary_group
    @brief      Determine the result value type when subtracting instances
                of @a Value2 from @a Value1.
    @sa         fern::algorithm::subtract::result_type
*/
template<
    typename Value1,
    typename Value2>
using result_value_type = typename fern::algorithm::Result<value_type<Value1>,
    value_type<Value2>>::type;


/*!
    @ingroup    fern_algorithm_algebra_elementary_group
    @brief      Out of range policy for fern::algorithm::algebra::subtract
                algorithm.
    @sa         @ref fern_algorithm_policies_out_of_range_policy

    The logic for determining whether subtract's result is out of range depends
    on the types involved (unsigned integers, signed integers, floating
    points) and their sizes.

    The value types of @a value1 and @a value2 must be arithmetic and not
    `bool`. The value type of @a result must be equal to
    fern::algorithm::subtract::result_value_type<Value1, Value2>.
*/
template<
    typename Value1,
    typename Value2,
    typename Result>
class OutOfRangePolicy
{

    FERN_STATIC_ASSERT(std::is_arithmetic, value_type<Value1>)
    FERN_STATIC_ASSERT(!std::is_same, value_type<Value1>, bool)
    FERN_STATIC_ASSERT(std::is_arithmetic, value_type<Value2>)
    FERN_STATIC_ASSERT(!std::is_same, value_type<Value2>, bool)
    FERN_STATIC_ASSERT(std::is_arithmetic, value_type<Result>)
    FERN_STATIC_ASSERT(std::is_same, value_type<Result>,
        subtract::result_value_type<Value1, Value2>)

public:

    inline static bool within_range(
        Value1 const& value1,
        Value2 const& value2,
        Result const& result)
    {
        using value1_tag = base_class<number_category<Value1>, integer_tag>;
        using value2_tag = base_class<number_category<Value2>, integer_tag>;

        return detail::dispatch::within_range<Value1, Value2, Result,
            value1_tag, value2_tag>::calculate(value1, value2, result);
    }

};


} // namespace subtract


namespace algebra {

/*!
    @ingroup    fern_algorithm_algebra_elementary_group
    @brief      Subtract @a value2 to @a value1 and write the result to
                @a result.
    @sa         fern::algorithm::subtract::OutOfRangePolicy,
                fern::algorithm::subtract::result_type,
                fern::algorithm::subtract::result_value_type,
                fern::algorithm::binary_local_operation

    The value types of @a value1 and @a value2 must be arithmetic and not
    `bool`. The value type of @a result must be equal to
    fern::algorithm::subtract::result_value_type<Value1, Value2>.
*/
template<
    template<typename, typename, typename> class OutOfRangePolicy,
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename ExecutionPolicy,
    typename Value1,
    typename Value2,
    typename Result
>
void subtract(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    ExecutionPolicy const& execution_policy,
    Value1 const& value1,
    Value2 const& value2,
    Result& result)
{
    FERN_STATIC_ASSERT(std::is_arithmetic, value_type<Value1>)
    FERN_STATIC_ASSERT(!std::is_same, value_type<Value1>, bool)
    FERN_STATIC_ASSERT(std::is_arithmetic, value_type<Value2>)
    FERN_STATIC_ASSERT(!std::is_same, value_type<Value2>, bool)
    FERN_STATIC_ASSERT(std::is_arithmetic, value_type<Result>)
    FERN_STATIC_ASSERT(std::is_same, value_type<Result>,
        subtract::result_value_type<Value1, Value2>)

    subtract::detail::subtract<OutOfRangePolicy>(input_no_data_policy,
        output_no_data_policy, execution_policy, value1, value2, result);
}


/*!
    @ingroup    fern_algorithm_algebra_elementary_group
    @overload
*/
template<
    typename ExecutionPolicy,
    typename Value1,
    typename Value2,
    typename Result
>
void subtract(
    ExecutionPolicy const& execution_policy,
    Value1 const& value1,
    Value2 const& value2,
    Result& result)
{
    using InputNoDataPolicy = InputNoDataPolicies<SkipNoData<>, SkipNoData<>>;
    using OutputNoDataPolicy = DontMarkNoData;

    OutputNoDataPolicy output_no_data_policy;
    subtract<binary::DiscardRangeErrors>(InputNoDataPolicy{{}, {}},
        output_no_data_policy, execution_policy, value1, value2, result);
}

} // namespace algebra
} // namespace algorithm
} // namespace fern

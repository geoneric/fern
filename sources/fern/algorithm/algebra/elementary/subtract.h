#pragma once
#include "fern/core/assert.h"
#include "fern/core/base_class.h"
#include "fern/algorithm/policy/policies.h"
#include "fern/algorithm/algebra/elementary/detail/subtract.h"


namespace fern {
namespace subtract {

//! Determine the result type when subtracting instances of \a Value2 from \a Value1.
/*!
    \sa            fern::subtract::result_value_type, fern::Result
*/
template<
    class Value1,
    class Value2>
using result_type = typename fern::Result<Value1, Value2>::type;


//! Determine the result value type when subtracting instances of \a Value2 from \a Value1.
/*!
  \sa              fern::subtract::result_type
*/
template<
    class Value1,
    class Value2>
using result_value_type = typename fern::Result<value_type<Value1>,
    value_type<Value2>>::type;


//! Out of range policy for fern::algebra::subtract algorithm.
/*!
    The logic for determining whether subtract's result is out of range depends
    on the types involved (unsigned integers, signed integers, floating
    points) and their sizes.

    The value types of \a value1 and \a value2 must be arithmetic and not
    `bool`. The value type of \a result must be equal to
    fern::subtract::result_value_type<Value1, Value2>.

    \sa            @ref fern_algorithm_policies_out_of_range_policy
*/
template<
    class Value1,
    class Value2,
    class Result>
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

//! Subtract \a value2 to \a value1 and write the result to \a result.
/*!
    \ingroup       elementary
    \sa            fern::subtract::OutOfRangePolicy,
                   fern::subtract::result_type,
                   fern::subtract::result_value_type,
                   fern::binary_local_operation,
                   @ref fern_algorithm_algebra_elementary

    The value types of \a value1 and \a value2 must be arithmetic and not
    `bool`. The value type of \a result must be equal to
    fern::subtract::result_value_type<Value1, Value2>.
*/
template<
    template<class, class, class> class OutOfRangePolicy,
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class ExecutionPolicy,
    class Value1,
    class Value2,
    class Result
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
    \ingroup       elementary
    \overload
*/
template<
    template<class, class, class> class OutOfRangePolicy,
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class ExecutionPolicy,
    class Value1,
    class Value2,
    class Result
>
void subtract(
    ExecutionPolicy const& execution_policy,
    Value1 const& value1,
    Value2 const& value2,
    Result& result)
{
    OutputNoDataPolicy output_no_data_policy;
    subtract<OutOfRangePolicy>(InputNoDataPolicy(), output_no_data_policy,
        execution_policy, value1, value2, result);
}


/*!
    \ingroup       elementary
    \overload
*/
template<
    class ExecutionPolicy,
    class Value1,
    class Value2,
    class Result
>
void subtract(
    ExecutionPolicy const& execution_policy,
    Value1 const& value1,
    Value2 const& value2,
    Result& result)
{
    using InputNoDataPolicy = SkipNoData;
    using OutputNoDataPolicy = DontMarkNoData;

    OutputNoDataPolicy output_no_data_policy;
    subtract<binary::DiscardRangeErrors>(InputNoDataPolicy(),
        output_no_data_policy, execution_policy, value1, value2, result);
}

} // namespace algebra
} // namespace fern

#pragma once
#include "fern/core/assert.h"
#include "fern/core/base_class.h"
#include "fern/algorithm/policy/policies.h"
#include "fern/algorithm/algebra/elementary/detail/absolute.h"


namespace fern {
namespace absolute {

//! Out of range policy for fern::algebra::absolute algorithm.
/*!
    The logic for determining whether absolute's result is out of range depends
    on the types involved (unsigned integers, signed integers, floating
    points) and their sizes.

    The value types of \a value1 and \a value2 must be arithmetic and not
    `bool`. The value type of \a result must be equal to
    fern::add::result_value_type<Value1, Value2>.

    \sa            @ref fern_algorithm_policies_out_of_range_policy
*/
template<
    class Value,
    class Result>
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

//! Determine the absolute value of \a value and write the result to \a result.
/*!
    \sa            fern::absolute::OutOfRangePolicy,
                   fern::unary_local_operation

    The value type of \a value must be arithmetic and not `bool`. The value
    type of \a result must be equal to the value type of \a value.
*/
template<
    template<class, class> class OutOfRangePolicy,
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class ExecutionPolicy,
    class Value,
    class Result
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
    \overload
*/
template<
    template<class, class> class OutOfRangePolicy,
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class ExecutionPolicy,
    class Value,
    class Result
>
void absolute(
    ExecutionPolicy const& execution_policy,
    Value const& value,
    Result& result)
{
    OutputNoDataPolicy output_no_data_policy;
    absolute<OutOfRangePolicy>(InputNoDataPolicy(), output_no_data_policy,
        execution_policy, value, result);
}


/*!
    \overload
*/
template<
    class ExecutionPolicy,
    class Value,
    class Result
>
void absolute(
    ExecutionPolicy const& execution_policy,
    Value const& value,
    Result& result)
{
    using InputNoDataPolicy = SkipNoData;
    using OutputNoDataPolicy = DontMarkNoData;

    OutputNoDataPolicy output_no_data_policy;
    absolute<unary::DiscardRangeErrors>(InputNoDataPolicy(),
        output_no_data_policy, execution_policy, value, result);
}

} // namespace algebra
} // namespace fern

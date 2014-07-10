#pragma once
#include "fern/core/base_class.h"
#include "fern/algorithm/policy/policies.h"
#include "fern/algorithm/algebra/elementary/detail/divide.h"


namespace fern {
namespace divide {

//! Determine the result type when dividing instances of \a Value1 with \a Value2.
/*!
    \sa            fern::divide::result_value_type, fern::Result
*/
template<
    class Value1,
    class Value2>
using result_type = typename fern::Result<Value1, Value2>::type;


//! Determine the result value type when dividing instances of \a Value1 with \a Value2.
/*!
  \sa              fern::divide::result_type
*/
template<
    class Value1,
    class Value2>
using result_value_type = typename fern::Result<value_type<Value1>,
    value_type<Value2>>::type;


//! Out of domain policy for fern::algebra::divide algorithm.
/*!
    The denominator (\a value2) cannot be zero. Otherwise all values are
    within the domain of valid values for divide.

    The value types of \a value1 and \a value2 must be arithmetic and not
    `bool`.

    \sa            @ref fern_algorithm_policies_out_of_domain_policy
*/
template<
    class Value1,
    class Value2>
class OutOfDomainPolicy
{

    FERN_STATIC_ASSERT(std::is_arithmetic, value_type<Value1>)
    FERN_STATIC_ASSERT(!std::is_same, value_type<Value1>, bool)
    FERN_STATIC_ASSERT(std::is_arithmetic, value_type<Value2>)
    FERN_STATIC_ASSERT(!std::is_same, value_type<Value2>, bool)

public:

    inline static bool within_domain(
        Value1 const& /* value1 */,
        Value2 const& value2)
    {
        return value2 != Value2(0);
    }

};


//! Out of range policy for fern::algebra::divide algorithm.
/*!
    The logic for determining whether divide's result is out of range depends
    on the types involved (unsigned integers, signed integers, floating
    points) and their sizes.

    The value types of \a value1 and \a value2 must be arithmetic and not
    `bool`. The value type of \a result must be equal to
    fern::divide::result_value_type<Value1, Value2>.

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
        divide::result_value_type<Value1, Value2>)

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

} // namespace divide


namespace algebra {

//! Divide \a value1 by \a value2 and write the result to \a result.
/*!
    \sa            fern::divide::OutOfDomainPolicy,
                   fern::divide::OutOfRangePolicy,
                   fern::divide::result_type,
                   fern::divide::result_value_type,
                   fern::binary_local_operation

    The value types of \a value1 and \a value2 must be arithmetic and not
    `bool`. The value type of \a result must be equal to
    fern::divide::result_value_type<Value1, Value2>.
*/
template<
    template<class, class> class OutOfDomainPolicy,
    template<class, class, class> class OutOfRangePolicy,
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class ExecutionPolicy,
    class Value1,
    class Value2,
    class Result
>
void divide(
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
        divide::result_value_type<Value1, Value2>)

    divide::detail::divide<OutOfDomainPolicy, OutOfRangePolicy>(
        input_no_data_policy, output_no_data_policy, execution_policy,
        value1, value2, result);
}


/*!
  \overload
*/
template<
    template<class, class> class OutOfDomainPolicy,
    template<class, class, class> class OutOfRangePolicy,
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class ExecutionPolicy,
    class Value1,
    class Value2,
    class Result
>
void divide(
    ExecutionPolicy const& execution_policy,
    Value1 const& value1,
    Value2 const& value2,
    Result& result)
{
    OutputNoDataPolicy output_no_data_policy;
    divide<OutOfRangePolicy>(InputNoDataPolicy(), output_no_data_policy,
        execution_policy, value1, value2, result);
}


/*!
  \overload
*/
template<
    class ExecutionPolicy,
    class Value1,
    class Value2,
    class Result
>
void divide(
    ExecutionPolicy const& execution_policy,
    Value1 const& value1,
    Value2 const& value2,
    Result& result)
{
    using InputNoDataPolicy = SkipNoData;
    using OutputNoDataPolicy = DontMarkNoData;

    OutputNoDataPolicy output_no_data_policy;
    divide<binary::DiscardDomainErrors, binary::DiscardRangeErrors>(
        InputNoDataPolicy(), output_no_data_policy, execution_policy,
        value1, value2, result);
}

} // namespace algebra
} // namespace fern

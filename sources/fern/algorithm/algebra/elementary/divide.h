#pragma once
#include "fern/core/base_class.h"
#include "fern/algorithm/policy/policies.h"
#include "fern/algorithm/algebra/elementary/detail/divide.h"


namespace fern {
namespace divide {

/*!
    The denominator cannot be zero. Otherwise all values are within the domain
    of valid values for divide.
*/
template<
    class Value1,
    class Value2>
class OutOfDomainPolicy
{
    FERN_STATIC_ASSERT(std::is_arithmetic, Value1)
    FERN_STATIC_ASSERT(std::is_arithmetic, Value2)

public:

    inline static bool within_domain(
        Value1 const& /* value1 */,
        Value2 const& value2)
    {
        return value2 != Value2(0);
    }

};


template<
    class Value1,
    class Value2,
    class Result>
class OutOfRangePolicy
{

    FERN_STATIC_ASSERT(std::is_arithmetic, Value1)
    FERN_STATIC_ASSERT(std::is_arithmetic, Value2)
    FERN_STATIC_ASSERT(std::is_arithmetic, Result)

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
    FERN_STATIC_ASSERT(std::is_arithmetic, value_type<Value2>)
    FERN_STATIC_ASSERT(std::is_arithmetic, value_type<Result>)
    FERN_STATIC_ASSERT(std::is_same, value_type<Result>,
        typename fern::Result<value_type<Value1>, value_type<Value2>>::type)

    divide::detail::divide<OutOfDomainPolicy, OutOfRangePolicy>(
        input_no_data_policy, output_no_data_policy, execution_policy,
        value1, value2, result);
}


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

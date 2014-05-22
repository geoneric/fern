#pragma once
#include "fern/core/base_class.h"
#include "fern/algorithm/core/binary_operation.h"
#include "fern/algorithm/core/operation_traits.h"
#include "fern/algorithm/policy/policies.h"
#include "fern/algorithm/algebra/elementary/detail/multiply.h"


namespace fern {
namespace multiply {

/*!
    All values are within the domain of valid values for multiply.
*/
template<
    class Value1,
    class Value2>
using OutOfDomainPolicy = DiscardDomainErrors<Value1, Value2>;


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

    inline bool within_range(
        Value1 const& value1,
        Value2 const& value2,
        Result const& result) const
    {
        using value1_tag = typename base_class<
            typename TypeTraits<Value1>::number_category, integer_tag>::type;
        using value2_tag = typename base_class<
            typename TypeTraits<Value2>::number_category, integer_tag>::type;

        return detail::dispatch::within_range<Value1, Value2, Result,
            value1_tag, value2_tag>::calculate(value1, value2, result);
    }

    OutOfRangePolicy()=default;

    ~OutOfRangePolicy()=default;

};


template<
    class Value1,
    class Value2>
struct Algorithm
{

    FERN_STATIC_ASSERT(std::is_arithmetic, Value1)
    FERN_STATIC_ASSERT(std::is_arithmetic, Value2)

    template<
        class R>
    inline void operator()(
        Value1 const& value1,
        Value2 const& value2,
        R& result) const
    {
        result = static_cast<R>(value1) * static_cast<R>(value2);
    }

};

} // namespace multiply


namespace algebra {

template<
    class Values1,
    class Values2,
    class Result,
    template<class, class> class OutOfDomainPolicy=binary::DiscardDomainErrors,
    template<class, class, class> class OutOfRangePolicy=
        binary::DiscardRangeErrors,
    class InputNoDataPolicy=SkipNoData,
    class OutputNoDataPolicy=DontMarkNoData
>
class Multiply
{

public:

    using category = local_operation_tag;
    using A1 = Values1;
    using A1Value = value_type<A1>;
    using A2 = Values2;
    using A2Value = value_type<A2>;
    using R = Result;
    using RValue = value_type<R>;

    FERN_STATIC_ASSERT(std::is_arithmetic, A1Value)
    FERN_STATIC_ASSERT(std::is_arithmetic, A2Value)
    FERN_STATIC_ASSERT(std::is_arithmetic, RValue)
    FERN_STATIC_ASSERT(std::is_same, RValue,
        typename fern::Result<A1Value, A2Value>::type)

    Multiply()
        : _algorithm(multiply::Algorithm<A1Value, A2Value>())
    {
    }

    Multiply(
        InputNoDataPolicy&& input_no_data_policy,  // Universal reference.
        OutputNoDataPolicy&& output_no_data_policy)  // Universal reference.
        : _algorithm(
            std::forward<InputNoDataPolicy>(input_no_data_policy),
            std::forward<OutputNoDataPolicy>(output_no_data_policy),
            multiply::Algorithm<A1Value, A2Value>())
    {
    }

    inline void operator()(
        A1 const& values1,
        A2 const& values2,
        R& result)
    {
        _algorithm.calculate(values1, values2, result);
    }

    template<
        class Indices>
    inline void operator()(
        Indices const& indices,
        A1 const& values1,
        A2 const& values2,
        R& result)
    {
        _algorithm.calculate(indices, values1, values2, result);
    }

private:

    detail::dispatch::BinaryOperation<A1, A2, R,
        OutOfDomainPolicy, OutOfRangePolicy, InputNoDataPolicy,
        OutputNoDataPolicy, multiply::Algorithm<A1Value, A2Value>,
        typename base_class<
            typename ArgumentTraits<A1>::argument_category,
            array_2d_tag>::type,
        typename base_class<
            typename ArgumentTraits<A2>::argument_category,
            array_2d_tag>::type> _algorithm;

};


//! Calculate the result of subtracting \a values2 from \a values1 and put it in \a result.
/*!
*/
template<
    class Values1,
    class Values2,
    class Result,
    template<class, class> class OutOfDomainPolicy=binary::DiscardDomainErrors,
    template<class, class, class> class OutOfRangePolicy=
        binary::DiscardRangeErrors,
    class InputNoDataPolicy=SkipNoData,
    class OutputNoDataPolicy=DontMarkNoData
>
void multiply(
    Values1 const& values1,
    Values2 const& values2,
    Result& result)
{
    Multiply<Values1, Values2, Result, OutOfDomainPolicy, OutOfRangePolicy,
        InputNoDataPolicy, OutputNoDataPolicy>()(values1, values2, result);
}


/*!
  \overload
*/
template<
    class Values1,
    class Values2,
    class Result,
    template<class, class> class OutOfDomainPolicy=binary::DiscardDomainErrors,
    template<class, class, class> class OutOfRangePolicy=
        binary::DiscardRangeErrors,
    class InputNoDataPolicy=SkipNoData,
    class OutputNoDataPolicy=DontMarkNoData
>
void multiply(
    InputNoDataPolicy&& input_no_data_policy,  // Universal reference.
    OutputNoDataPolicy&& output_no_data_policy,  // Universal reference.
    Values1 const& values1,
    Values2 const& values2,
    Result& result)
{
    Multiply<Values1, Values2, Result, OutOfDomainPolicy, OutOfRangePolicy,
        InputNoDataPolicy, OutputNoDataPolicy>(
            std::forward<InputNoDataPolicy>(input_no_data_policy),
            std::forward<OutputNoDataPolicy>(output_no_data_policy))(
        values1, values2, result);
}

} // namespace algebra
} // namespace fern

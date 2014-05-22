#pragma once
#include "fern/core/assert.h"
#include "fern/core/base_class.h"
#include "fern/algorithm/core/operation_traits.h"
#include "fern/algorithm/core/unary_operation.h"
#include "fern/algorithm/policy/policies.h"
#include "fern/algorithm/algebra/elementary/detail/absolute.h"


namespace fern {
namespace absolute {

// All values are within the domain of valid values for absolute.
template<
    class Value>
using OutOfDomainPolicy = DiscardDomainErrors<Value>;


template<
    class Value,
    class Result>
class OutOfRangePolicy
{

    FERN_STATIC_ASSERT(std::is_arithmetic, Value)
    FERN_STATIC_ASSERT(std::is_arithmetic, Result)

public:

    inline bool within_range(
        Value const& value,
        Result const& result) const
    {
        using value_tag = typename base_class<
            typename TypeTraits<Value>::number_category, integer_tag>::type;

        return detail::dispatch::within_range<Value, Result, value_tag>::
            calculate(value, result);
    }

};


template<
    class Value>
struct Algorithm
{

    FERN_STATIC_ASSERT(std::is_arithmetic, Value)

    template<
        class R>
    inline void operator()(
        Value const& value,
        R& result) const
    {
        result = static_cast<R>(std::abs(value));
    }

};

} // namespace absolute


namespace algebra {

template<
    class Values,
    class Result,
    template<class> class OutOfDomainPolicy=unary::DiscardDomainErrors,
    template<class, class> class OutOfRangePolicy=unary::DiscardRangeErrors,
    class InputNoDataPolicy=SkipNoData,
    class OutputNoDataPolicy=DontMarkNoData
>
class Absolute
{

public:

    using category = local_operation_tag;
    using A = Values;
    using AValue = value_type<A>;
    using R = Result;
    using RValue = value_type<R>;

    FERN_STATIC_ASSERT(std::is_arithmetic, AValue)
    FERN_STATIC_ASSERT(std::is_arithmetic, RValue)

    Absolute()
        : _algorithm(absolute::Algorithm<AValue>())
    {
    }

    Absolute(
        InputNoDataPolicy&& input_no_data_policy,  // Universal reference.
        OutputNoDataPolicy&& output_no_data_policy)  // Universal reference.
        : _algorithm(
            std::forward<InputNoDataPolicy>(input_no_data_policy),
            std::forward<OutputNoDataPolicy>(output_no_data_policy),
            absolute::Algorithm<AValue>())
    {
    }

    inline void operator()(
        A const& values,
        R& result)
    {
        _algorithm.calculate(values, result);
    }

    template<
        class Indices>
    inline void operator()(
        Indices const& indices,
        A const& values,
        R& result)
    {
        _algorithm.calculate(indices, values, result);
    }

private:

    detail::dispatch::UnaryOperation<A, R,
        OutOfDomainPolicy, OutOfRangePolicy, InputNoDataPolicy,
        OutputNoDataPolicy, absolute::Algorithm<AValue>,
        typename base_class<
            typename ArgumentTraits<A>::argument_category,
            array_2d_tag>::type> _algorithm;

};


//! Calculate the absolute value of each value in \a values store the result in \a result.
/*!
*/
template<
    class Value,
    class Result,
    template<class> class OutOfDomainPolicy=unary::DiscardDomainErrors,
    template<class, class> class OutOfRangePolicy=unary::DiscardRangeErrors,
    class InputNoDataPolicy=SkipNoData,
    class OutputNoDataPolicy=DontMarkNoData
>
void absolute(
    Value const& values,
    Result& result)
{
    Absolute<Value, Result, OutOfDomainPolicy, OutOfRangePolicy,
        InputNoDataPolicy, OutputNoDataPolicy>()(values, result);
}


/*!
  \overload
*/
template<
    class Value,
    class Result,
    template<class> class OutOfDomainPolicy=unary::DiscardDomainErrors,
    template<class, class> class OutOfRangePolicy=unary::DiscardRangeErrors,
    class InputNoDataPolicy=SkipNoData,
    class OutputNoDataPolicy=DontMarkNoData
>
void absolute(
    InputNoDataPolicy&& input_no_data_policy,  // Universal reference.
    OutputNoDataPolicy&& output_no_data_policy,  // Universal reference.
    Value const& values,
    Result& result)
{
    Absolute<Value, Result, OutOfDomainPolicy, OutOfRangePolicy,
        InputNoDataPolicy, OutputNoDataPolicy>(
            std::forward<InputNoDataPolicy>(input_no_data_policy),
            std::forward<OutputNoDataPolicy>(output_no_data_policy))(
        values, result);
}

} // namespace algebra
} // namespace fern

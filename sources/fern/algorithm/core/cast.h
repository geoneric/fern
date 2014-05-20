#pragma once
#include "fern/core/assert.h"
#include "fern/core/base_class.h"
#include "fern/algorithm/core/operation_traits.h"
#include "fern/algorithm/policy/policies.h"
#include "fern/algorithm/algebra/unary_operation.h"
#include "fern/algorithm/core/detail/cast.h"


namespace fern {
namespace cast {

// All values are within the domain of valid values for cast.
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
        using result_tag = typename base_class<
            typename TypeTraits<Result>::number_category, integer_tag>::type;

        return detail::dispatch::within_range<Value, Result, value_tag,
            result_tag>::calculate(value, result);
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
        result = static_cast<R>(value);
    }

};

} // namespace cast


namespace core {

template<
    class Values,
    class Result,
    template<class> class OutOfDomainPolicy=unary::DiscardDomainErrors,
    template<class, class> class OutOfRangePolicy=unary::DiscardRangeErrors,
    class InputNoDataPolicy=SkipNoData,
    class OutputNoDataPolicy=DontMarkNoData
>
class Cast
{

public:

    using category = local_operation_tag;
    using A = Values;
    using AValue = value_type<A>;
    using R = Result;
    using RValue = value_type<R>;

    FERN_STATIC_ASSERT(std::is_arithmetic, AValue)
    FERN_STATIC_ASSERT(std::is_arithmetic, RValue)

    Cast()
        : _algorithm(cast::Algorithm<AValue>())
    {
    }

    Cast(
        InputNoDataPolicy&& input_no_data_policy,  // Universal reference.
        OutputNoDataPolicy&& output_no_data_policy)  // Universal reference.
        : _algorithm(
            std::forward<InputNoDataPolicy>(input_no_data_policy),
            std::forward<OutputNoDataPolicy>(output_no_data_policy),
            cast::Algorithm<AValue>())
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
        OutputNoDataPolicy, cast::Algorithm<AValue>,
        typename base_class<
            typename ArgumentTraits<A>::argument_category,
            array_2d_tag>::type> _algorithm;

};


template<
    class Value,
    class Result,
    template<class> class OutOfDomainPolicy=unary::DiscardDomainErrors,
    template<class, class> class OutOfRangePolicy=unary::DiscardRangeErrors,
    class InputNoDataPolicy=SkipNoData,
    class OutputNoDataPolicy=DontMarkNoData
>
void cast(
    Value const& values,
    Result& result)
{
    Cast<Value, Result, OutOfDomainPolicy, OutOfRangePolicy,
        InputNoDataPolicy, OutputNoDataPolicy>()(values, result);
}


template<
    class Value,
    class Result,
    template<class> class OutOfDomainPolicy=unary::DiscardDomainErrors,
    template<class, class> class OutOfRangePolicy=unary::DiscardRangeErrors,
    class InputNoDataPolicy=SkipNoData,
    class OutputNoDataPolicy=DontMarkNoData
>
void cast(
    InputNoDataPolicy&& input_no_data_policy,  // Universal reference.
    OutputNoDataPolicy&& output_no_data_policy,  // Universal reference.
    Value const& values,
    Result& result)
{
    Cast<Value, Result, OutOfDomainPolicy, OutOfRangePolicy,
        InputNoDataPolicy, OutputNoDataPolicy>(
            std::forward<InputNoDataPolicy>(input_no_data_policy),
            std::forward<OutputNoDataPolicy>(output_no_data_policy))(
        values, result);
}

} // namespace core
} // namespace fern

#pragma once
#include <cmath>
#include "fern/core/assert.h"
#include "fern/core/base_class.h"
#include "fern/algorithm/core/operation_traits.h"
#include "fern/algorithm/core/unary_local_operation.h"
#include "fern/algorithm/policy/policies.h"


namespace fern {
namespace tan {

template<
    class Value
>
class OutOfDomainPolicy
{

    FERN_STATIC_ASSERT(std::is_floating_point, Value)

public:

    inline bool within_domain(
        Value const& value) const
    {
        // - value must be finite and
        // - value must not be divisable by an odd number of times 0.5 * pi.

        if(!std::isfinite(value)) {
            // All bets are off.
            return false;
        }
        else {
            Value remainder = std::remainder(value, fern::half_pi<Value>());

            if(remainder != Value(0)) {
                // Value is not divisable by a whole number of times 0.5 * pi.
                return true;
            }
            else {
                Value quotient = value / fern::half_pi<Value>();

                return int64_t(quotient) % 2 == 0;
            }
        }
    }

};


template<
    class Value>
struct Algorithm
{

    FERN_STATIC_ASSERT(std::is_floating_point, Value)

    template<
        class Result>
    inline void operator()(
        Value const& value,
        Result& result) const
    {
        FERN_STATIC_ASSERT(std::is_same, Result, Value)

        result = std::tan(value);
    }

};

} // namespace tan


namespace trigonometry {

template<
    class Values,
    class Result,
    template<class> class OutOfDomainPolicy=unary::DiscardDomainErrors,
    class InputNoDataPolicy=SkipNoData,
    class OutputNoDataPolicy=DontMarkNoData
>
class Tan
{

public:

    using category = local_operation_tag;
    using A = Values;
    using AValue = value_type<A>;
    using R = Result;
    using RValue = value_type<R>;

    FERN_STATIC_ASSERT(std::is_floating_point, AValue)
    FERN_STATIC_ASSERT(std::is_same, RValue, AValue)

    Tan()
        : _algorithm(tan::Algorithm<AValue>())
    {
    }

    Tan(
        InputNoDataPolicy&& input_no_data_policy,  // Universal reference.
        OutputNoDataPolicy&& output_no_data_policy)  // Universal reference.
        : _algorithm(
            std::forward<InputNoDataPolicy>(input_no_data_policy),
            std::forward<OutputNoDataPolicy>(output_no_data_policy),
            tan::Algorithm<AValue>())
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

    detail::dispatch::UnaryLocalOperation<A, R,
        OutOfDomainPolicy, unary::DiscardRangeErrors,
        InputNoDataPolicy, OutputNoDataPolicy, tan::Algorithm<AValue>,
        typename base_class<
            typename ArgumentTraits<A>::argument_category,
            array_2d_tag>::type> _algorithm;

};


template<
    class Value,
    class Result,
    template<class> class OutOfDomainPolicy=unary::DiscardDomainErrors,
    class InputNoDataPolicy=SkipNoData,
    class OutputNoDataPolicy=DontMarkNoData
>
void tan(
    Value const& values,
    Result& result)
{
    Tan<Value, Result, OutOfDomainPolicy,
        InputNoDataPolicy, OutputNoDataPolicy>()(values, result);
}


template<
    class Value,
    class Result,
    template<class> class OutOfDomainPolicy=unary::DiscardDomainErrors,
    class InputNoDataPolicy=SkipNoData,
    class OutputNoDataPolicy=DontMarkNoData
>
void tan(
    InputNoDataPolicy&& input_no_data_policy,  // Universal reference.
    OutputNoDataPolicy&& output_no_data_policy,  // Universal reference.
    Value const& values,
    Result& result)
{
    Tan<Value, Result, OutOfDomainPolicy,
        InputNoDataPolicy, OutputNoDataPolicy>(
            std::forward<InputNoDataPolicy>(input_no_data_policy),
            std::forward<OutputNoDataPolicy>(output_no_data_policy))(
        values, result);
}

} // namespace trigonometry
} // namespace fern

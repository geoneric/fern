#pragma once
#include <cmath>
#include "fern/core/assert.h"
#include "fern/core/base_class.h"
#include "fern/algorithm/core/operation_traits.h"
#include "fern/algorithm/core/unary_local_operation.h"
#include "fern/algorithm/policy/policies.h"


namespace fern {
namespace atan {

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

        result = std::atan(value);
    }

};

} // namespace atan


namespace trigonometry {

template<
    class Values,
    class Result,
    class InputNoDataPolicy=SkipNoData,
    class OutputNoDataPolicy=DontMarkNoData
>
class ATan
{

public:

    using category = local_operation_tag;
    using A = Values;
    using AValue = value_type<A>;
    using R = Result;
    using RValue = value_type<R>;

    FERN_STATIC_ASSERT(std::is_floating_point, AValue)
    FERN_STATIC_ASSERT(std::is_same, RValue, AValue)

    ATan()
        : _algorithm(atan::Algorithm<AValue>())
    {
    }

    ATan(
        InputNoDataPolicy&& input_no_data_policy,  // Universal reference.
        OutputNoDataPolicy&& output_no_data_policy)  // Universal reference.
        : _algorithm(
            std::forward<InputNoDataPolicy>(input_no_data_policy),
            std::forward<OutputNoDataPolicy>(output_no_data_policy),
            atan::Algorithm<AValue>())
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
        unary::DiscardDomainErrors, unary::DiscardRangeErrors,
        InputNoDataPolicy, OutputNoDataPolicy, atan::Algorithm<AValue>,
        base_class<argument_category<A>, array_2d_tag>> _algorithm;

};


template<
    class Value,
    class Result,
    class InputNoDataPolicy=SkipNoData,
    class OutputNoDataPolicy=DontMarkNoData
>
void atan(
    Value const& values,
    Result& result)
{
    ATan<Value, Result,
        InputNoDataPolicy, OutputNoDataPolicy>()(values, result);
}


template<
    class Value,
    class Result,
    class InputNoDataPolicy=SkipNoData,
    class OutputNoDataPolicy=DontMarkNoData
>
void atan(
    InputNoDataPolicy&& input_no_data_policy,  // Universal reference.
    OutputNoDataPolicy&& output_no_data_policy,  // Universal reference.
    Value const& values,
    Result& result)
{
    ATan<Value, Result,
        InputNoDataPolicy, OutputNoDataPolicy>(
            std::forward<InputNoDataPolicy>(input_no_data_policy),
            std::forward<OutputNoDataPolicy>(output_no_data_policy))(
        values, result);
}

} // namespace trigonometry
} // namespace fern

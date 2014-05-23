#pragma once
#include "fern/algorithm/core/operation_traits.h"
#include "fern/algorithm/policy/policies.h"
#include "fern/algorithm/statistic/detail/unary_max.h"


namespace fern {
namespace statistic {

template<
    class Values,
    class Result,
    class InputNoDataPolicy=SkipNoData,
    class OutputNoDataPolicy=DontMarkNoData
>
class UnaryMax
{

public:

    using category = local_aggregate_operation_tag;
    using A = Values;
    using AValue = value_type<A>;
    using R = Result;
    using RValue = value_type<R>;

    FERN_STATIC_ASSERT(std::is_arithmetic, AValue)
    // We see boolean more as nominal values than as integers. You can't max'em.
    FERN_STATIC_ASSERT(!std::is_same, AValue, bool)
    FERN_STATIC_ASSERT(std::is_same, R, RValue)
    FERN_STATIC_ASSERT(std::is_same, RValue, AValue)

    UnaryMax()
        : _algorithm()
    {
    }

    UnaryMax(
        InputNoDataPolicy&& input_no_data_policy,  // Universal reference.
        OutputNoDataPolicy&& output_no_data_policy)  // Universal reference.
        : _algorithm(
              std::forward<InputNoDataPolicy>(input_no_data_policy),
              std::forward<OutputNoDataPolicy>(output_no_data_policy))
    {
    }

    inline void operator()(
        A const& values,
        R & result)
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


    template<
        class InputNoDataPolicy_,
        class Collection>
    inline void aggregate(
        InputNoDataPolicy_&& input_no_data_policy,  // Universal reverence.
        Collection const& results,
        R& result)
    {
        UnaryMax<Collection, R, InputNoDataPolicy_, OutputNoDataPolicy>(
            std::forward<InputNoDataPolicy_>(input_no_data_policy),
            std::forward<OutputNoDataPolicy>(_algorithm))(results, result);
    }

private:

    unary_max::detail::dispatch::UnaryMax<A, R,
        InputNoDataPolicy, OutputNoDataPolicy,
        typename ArgumentTraits<A>::argument_category> _algorithm;

};


template<
    class Values,
    class Result,
    class InputNoDataPolicy=SkipNoData,
    class OutputNoDataPolicy=DontMarkNoData
>
void unary_max(
    Values const& values,
    Result& result)
{
    UnaryMax<Values, Result, InputNoDataPolicy, OutputNoDataPolicy>()(
        values, result);
}


template<
    class Values,
    class Result,
    class InputNoDataPolicy=SkipNoData,
    class OutputNoDataPolicy=DontMarkNoData
>
void unary_max(
    InputNoDataPolicy&& input_no_data_policy,  // Universal reference.
    OutputNoDataPolicy&& output_no_data_policy,  // Universal reference.
    Values const& values,
    Result& result)
{
    UnaryMax<Values, Result, InputNoDataPolicy, OutputNoDataPolicy>(
        std::forward<InputNoDataPolicy>(input_no_data_policy),
        std::forward<OutputNoDataPolicy>(output_no_data_policy))(
            values, result);
}

} // namespace statistic
} // namespace fern

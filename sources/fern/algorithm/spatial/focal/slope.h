#pragma once
#include "fern/algorithm/core/operation_traits.h"
#include "fern/algorithm/policy/policies.h"
#include "fern/algorithm/spatial/focal/detail/slope.h"


namespace fern {
namespace slope {

template<
    class Result>
class OutOfRangePolicy
{

    FERN_STATIC_ASSERT(std::is_arithmetic, Result)
    FERN_STATIC_ASSERT(std::is_floating_point, Result)

public:

    inline bool within_range(
        Result const& result) const
    {
        return std::isfinite(result);
    }

};

} // namespace slope

namespace spatial {

template<
    class Values,
    class Result,
    class InputNoDataPolicy=SkipNoData,
    class OutputNoDataPolicy=DontMarkNoData
>
class Slope
{

public:

    using category = focal_operation_tag;
    using A = Values;
    using AValue = value_type<A>;
    using R = Result;
    using RValue = value_type<R>;

    FERN_STATIC_ASSERT(std::is_floating_point, AValue)
    FERN_STATIC_ASSERT(std::is_same, RValue, AValue)

    Slope()
        : _algorithm()
    {
    }

    Slope(
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
        Slope<Collection, R, InputNoDataPolicy_, OutputNoDataPolicy>(
            std::forward<InputNoDataPolicy_>(input_no_data_policy),
            std::forward<OutputNoDataPolicy>(_algorithm))(results, result);
    }

private:

    slope::detail::dispatch::Slope<A, R,
        InputNoDataPolicy, OutputNoDataPolicy,
        typename ArgumentTraits<A>::argument_category> _algorithm;

};


template<
    class Values,
    class Result,
    class InputNoDataPolicy=SkipNoData,
    class OutputNoDataPolicy=DontMarkNoData
>
void slope(
    Values const& values,
    Result& result)
{
    Slope<Values, Result, InputNoDataPolicy, OutputNoDataPolicy>()(
        values, result);
}


template<
    class Values,
    class Result,
    class InputNoDataPolicy=SkipNoData,
    class OutputNoDataPolicy=DontMarkNoData
>
void slope(
    InputNoDataPolicy&& input_no_data_policy,  // Universal reference.
    OutputNoDataPolicy&& output_no_data_policy,  // Universal reference.
    Values const& values,
    Result& result)
{
    Slope<Values, Result, InputNoDataPolicy, OutputNoDataPolicy>(
        std::forward<InputNoDataPolicy>(input_no_data_policy),
        std::forward<OutputNoDataPolicy>(output_no_data_policy))(
            values, result);
}

} // namespace spatial
} // namespace fern

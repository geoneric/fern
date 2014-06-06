#pragma once
#include "fern/core/assert.h"
#include "fern/core/base_class.h"
#include "fern/algorithm/core/operation_traits.h"
#include "fern/algorithm/core/binary_local_operation.h"
#include "fern/algorithm/policy/policies.h"


namespace fern {
namespace less_equal {

template<
    class Value1,
    class Value2,
    class Result>
struct Algorithm
{

    inline void operator()(
        Value1 const& value1,
        Value2 const& value2,
        Result& result) const
    {
        result = value1 <= value2;
    }

};

} // namespace less_equal


namespace algebra {

template<
    class Values1,
    class Values2,
    class Result,
    class InputNoDataPolicy=SkipNoData,
    class OutputNoDataPolicy=DontMarkNoData
>
class LessEqual
{

public:

    using category = local_operation_tag;
    using A1 = Values1;
    using A1Value = typename ArgumentTraits<A1>::value_type;
    using A1ConstReference = typename ArgumentTraits<A1>::const_reference;
    using A2 = Values2;
    using A2Value = typename ArgumentTraits<A2>::value_type;
    using A2ConstReference = typename ArgumentTraits<A2>::const_reference;
    using R = Result;
    using RValue = typename ArgumentTraits<R>::value_type;
    using RReference = typename ArgumentTraits<R>::reference;

    FERN_STATIC_ASSERT(std::is_arithmetic, A1Value)
    FERN_STATIC_ASSERT(!std::is_same, A1Value, bool)
    FERN_STATIC_ASSERT(std::is_arithmetic, A2Value)
    FERN_STATIC_ASSERT(!std::is_same, A2Value, bool)
    FERN_STATIC_ASSERT(std::is_arithmetic, RValue)
    FERN_STATIC_ASSERT(std::is_same, RValue, bool)

    LessEqual()
        : _algorithm(less_equal::Algorithm<A1ConstReference,
              A2ConstReference, RReference>())
    {
    }

    LessEqual(
        InputNoDataPolicy&& input_no_data_policy,  // Universal reference.
        OutputNoDataPolicy&& output_no_data_policy)  // Universal reference.
        : _algorithm(
            std::forward<InputNoDataPolicy>(input_no_data_policy),
            std::forward<OutputNoDataPolicy>(output_no_data_policy),
            less_equal::Algorithm<A1ConstReference, A2ConstReference,
                RReference>())
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

    detail::dispatch::BinaryLocalOperation<A1, A2, R,
        binary::DiscardDomainErrors, binary::DiscardRangeErrors,
        InputNoDataPolicy, OutputNoDataPolicy, less_equal::Algorithm<
            A1ConstReference, A2ConstReference, RReference>,
        base_class<argument_category<A1>, array_2d_tag>,
        base_class<argument_category<A2>, array_2d_tag>> _algorithm;

};


template<
    class Values1,
    class Values2,
    class Result,
    class InputNoDataPolicy=SkipNoData,
    class OutputNoDataPolicy=DontMarkNoData
>
void less_equal(
    Values1 const& values1,
    Values2 const& values2,
    Result& result)
{
    LessEqual<Values1, Values2, Result, InputNoDataPolicy,
        OutputNoDataPolicy>()(values1, values2, result);
}


template<
    class Values1,
    class Values2,
    class Result,
    class InputNoDataPolicy=SkipNoData,
    class OutputNoDataPolicy=DontMarkNoData
>
void less_equal(
    InputNoDataPolicy&& input_no_data_policy,  // Universal reference.
    OutputNoDataPolicy&& output_no_data_policy,  // Universal reference.
    Values1 const& values1,
    Values2 const& values2,
    Result& result)
{
    LessEqual<Values1, Values2, Result, InputNoDataPolicy,
        OutputNoDataPolicy>(
            std::forward<InputNoDataPolicy>(input_no_data_policy),
            std::forward<OutputNoDataPolicy>(output_no_data_policy))(
        values1, values2, result);
}

} // namespace algebra
} // namespace fern

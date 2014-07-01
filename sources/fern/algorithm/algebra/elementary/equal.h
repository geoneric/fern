#pragma once
#include "fern/core/argument_traits.h"
#include "fern/core/assert.h"
#include "fern/algorithm/core/binary_local_operation.h"
#include "fern/algorithm/core/operation_traits.h"
#include "fern/algorithm/policy/dont_mark_no_data.h"
#include "fern/algorithm/policy/discard_domain_errors.h"
#include "fern/algorithm/policy/discard_range_errors.h"
#include "fern/algorithm/policy/skip_no_data.h"
#include "fern/algorithm/algebra/result_type.h"


namespace fern {
namespace equal {

// All values are within the domain of valid values for equal.
template<
    class Value1,
    class Value2>
using OutOfDomainPolicy = DiscardDomainErrors<Value1, Value2>;


// All values are within the range of valid values for equal (given the bool
// output value type and the algorithm).
template<
    class Value1,
    class Value2>
using OutOfRangePolicy = DiscardRangeErrors<Value1, Value2>;


template<
    class Values1,
    class Values2,
    class Result>
struct Algorithm
{

    inline void operator()(
        Values1 values1,
        Values2 values2,
        Result result) const
    {
        result = values1 == values2;
    }

};

} // namespace equal


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
class Equal
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
    FERN_STATIC_ASSERT(std::is_arithmetic, A2Value)
    FERN_STATIC_ASSERT(std::is_arithmetic, RValue)
    FERN_STATIC_ASSERT(std::is_same, RValue, bool)

    Equal()
        : _algorithm(equal::Algorithm<A1ConstReference, A2ConstReference,
              RReference>())
    {
    }

    Equal(
        InputNoDataPolicy&& input_no_data_policy,  // Universal reference.
        OutputNoDataPolicy&& output_no_data_policy)  // Universal reference.
        : _algorithm(
            std::forward<InputNoDataPolicy>(input_no_data_policy),
            std::forward<OutputNoDataPolicy>(output_no_data_policy),
            equal::Algorithm<A1ConstReference, A2ConstReference, RReference>())
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
        OutOfDomainPolicy, OutOfRangePolicy, InputNoDataPolicy,
        OutputNoDataPolicy, equal::Algorithm<A1ConstReference,
            A2ConstReference, RReference>,
        base_class<argument_category<A1>, array_2d_tag>,
        base_class<argument_category<A2>, array_2d_tag>> _algorithm;

};


//! Calculate the result of comparing \a values1 to \a values2 and put it in \a result.
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
void equal(
    Values1 const& values1,
    Values2 const& values2,
    Result& result)
{
    Equal<Values1, Values2, Result, OutOfDomainPolicy, OutOfRangePolicy,
        InputNoDataPolicy, OutputNoDataPolicy>()(values1, values2, result);
}


//! Calculate the result of comparing \a values1 to \a values2 and put it in \a result.
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
void equal(
    InputNoDataPolicy&& input_no_data_policy,  // Universal reference.
    OutputNoDataPolicy&& output_no_data_policy,  // Universal reference.
    Values1 const& values1,
    Values2 const& values2,
    Result& result)
{
    Equal<Values1, Values2, Result, OutOfDomainPolicy, OutOfRangePolicy,
        InputNoDataPolicy, OutputNoDataPolicy>(
            std::forward<InputNoDataPolicy>(input_no_data_policy),
            std::forward<OutputNoDataPolicy>(output_no_data_policy))(
        values1, values2, result);
}

} // namespace algebra
} // namespace fern

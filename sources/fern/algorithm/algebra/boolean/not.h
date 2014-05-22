#pragma once
#include "fern/core/assert.h"
#include "fern/core/base_class.h"
#include "fern/algorithm/core/operation_traits.h"
#include "fern/algorithm/policy/policies.h"
#include "fern/algorithm/algebra/unary_operation.h"


namespace fern {
namespace not_ {

// All values are within the domain of valid values for not.
template<
    class Value>
using OutOfDomainPolicy = DiscardDomainErrors<Value>;


// All result values are within the range of valid values for not.
template<
    class Value,
    class Result>
using OutOfRangePolicy = DiscardRangeErrors<Value, Result>;


template<
    class Value>
struct Algorithm
{

    FERN_STATIC_ASSERT(std::is_same, Value, bool)

    template<
        class R>
    inline void operator()(
        Value const& value,
        R& result) const
    {
        result = !value;
    }

};

} // namespace not


namespace algebra {

template<
    class Values,
    class Result,
    template<class> class OutOfDomainPolicy=unary::DiscardDomainErrors,
    template<class, class> class OutOfRangePolicy=unary::DiscardRangeErrors,
    class InputNoDataPolicy=SkipNoData,
    class OutputNoDataPolicy=DontMarkNoData
>
class Not
{

public:

    using category = local_operation_tag;
    using A = Values;
    using AValue = value_type<A>;
    using R = Result;
    using RValue = value_type<R>;

    FERN_STATIC_ASSERT(std::is_same, AValue, bool)
    FERN_STATIC_ASSERT(std::is_same, RValue, bool)

    Not()
        : _algorithm(not_::Algorithm<AValue>())
    {
    }

    Not(
        InputNoDataPolicy&& input_no_data_policy,  // Universal reference.
        OutputNoDataPolicy&& output_no_data_policy)  // Universal reference.
        : _algorithm(
            std::forward<InputNoDataPolicy>(input_no_data_policy),
            std::forward<OutputNoDataPolicy>(output_no_data_policy),
            not_::Algorithm<AValue>())
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
        OutputNoDataPolicy, not_::Algorithm<AValue>,
        typename base_class<
            typename ArgumentTraits<A>::argument_category,
            array_2d_tag>::type> _algorithm;

};


//! Determine which values are valid.
/*!
*/
template<
    class Values,
    class Result,
    template<class> class OutOfDomainPolicy=unary::DiscardDomainErrors,
    template<class, class> class OutOfRangePolicy=unary::DiscardRangeErrors,
    class InputNoDataPolicy=SkipNoData,
    class OutputNoDataPolicy=DontMarkNoData
>
void not_(
    Values const& values,
    Result& result)
{
    Not<Values, Result, OutOfDomainPolicy, OutOfRangePolicy,
        InputNoDataPolicy, OutputNoDataPolicy>()(values, result);
}


/*!
  \overload
*/
template<
    class Values,
    class Result,
    template<class> class OutOfDomainPolicy=unary::DiscardDomainErrors,
    template<class, class> class OutOfRangePolicy=unary::DiscardRangeErrors,
    class InputNoDataPolicy=SkipNoData,
    class OutputNoDataPolicy=DontMarkNoData
>
void not_(
    InputNoDataPolicy&& input_no_data_policy,  // Universal reference.
    OutputNoDataPolicy&& output_no_data_policy,  // Universal reference.
    Values const& values,
    Result& result)
{
    Not<Values, Result, OutOfDomainPolicy, OutOfRangePolicy,
        InputNoDataPolicy, OutputNoDataPolicy>(
            std::forward<InputNoDataPolicy>(input_no_data_policy),
            std::forward<OutputNoDataPolicy>(output_no_data_policy))(
        values, result);
}

} // namespace algebra
} // namespace fern

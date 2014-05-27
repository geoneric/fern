#pragma once
#include <cmath>
#include "fern/core/base_class.h"
#include "fern/algorithm/core/binary_local_operation.h"
#include "fern/algorithm/core/operation_traits.h"
#include "fern/algorithm/policy/policies.h"


namespace fern {
namespace pow {

template<
    class Base,
    class Exponent
>
class OutOfDomainPolicy
{

public:

    inline bool within_domain(
        Base const& base,
        Exponent const& exponent) const
    {
        if(base < Base(0)) {
            Base integral, fractional;
            fractional = std::modf(exponent, &integral);

            if(fractional != Base(0)) {
                return false;
            }
        }
        else if(base == Base(0) && exponent < Exponent(0)) {
            return false;
        }

        return true;
    }

};


template<
    class Value1,
    class Value2,
    class Result>
using OutOfRangePolicy = DetectOutOfRangeByErrno<Value1, Value2, Result>;


template<
    class Value1,
    class Value2,
    class Result>
struct Algorithm
{

    FERN_STATIC_ASSERT(std::is_floating_point, Value1)
    FERN_STATIC_ASSERT(std::is_same, Value2, Value1)

    inline void operator()(
        Value1 const& value1,
        Value2 const& value2,
        Result& result) const
    {
        errno = 0;
        result = std::pow(static_cast<Result>(value1),
            static_cast<Result>(value2));
    }

};

} // namespace pow


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
class Pow
{

public:

    using category = local_operation_tag;
    using A1 = Values1;
    using A1Value = value_type<A1>;
    using A2 = Values2;
    using A2Value = value_type<A2>;
    using R = Result;
    using RValue = value_type<R>;

    FERN_STATIC_ASSERT(std::is_floating_point, A1Value)
    FERN_STATIC_ASSERT(std::is_same, A2Value, A1Value)
    FERN_STATIC_ASSERT(std::is_same, RValue, A1Value)

    Pow()
        : _algorithm(pow::Algorithm<A1Value, A2Value, RValue>())
    {
    }

    Pow(
        InputNoDataPolicy&& input_no_data_policy,  // Universal reference.
        OutputNoDataPolicy&& output_no_data_policy)  // Universal reference.
        : _algorithm(
            std::forward<InputNoDataPolicy>(input_no_data_policy),
            std::forward<OutputNoDataPolicy>(output_no_data_policy),
            pow::Algorithm<A1Value, A2Value, RValue>())
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
        OutputNoDataPolicy, pow::Algorithm<A1Value, A2Value, RValue>,
        typename base_class<
            typename ArgumentTraits<A1>::argument_category,
            array_2d_tag>::type,
        typename base_class<
            typename ArgumentTraits<A2>::argument_category,
            array_2d_tag>::type> _algorithm;

};


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
void pow(
    Values1 const& values1,
    Values2 const& values2,
    Result& result)
{
    Pow<Values1, Values2, Result, OutOfDomainPolicy, OutOfRangePolicy,
        InputNoDataPolicy, OutputNoDataPolicy>()(values1, values2, result);
}


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
void pow(
    InputNoDataPolicy&& input_no_data_policy,  // Universal reference.
    OutputNoDataPolicy&& output_no_data_policy,  // Universal reference.
    Values1 const& values1,
    Values2 const& values2,
    Result& result)
{
    Pow<Values1, Values2, Result, OutOfDomainPolicy, OutOfRangePolicy,
        InputNoDataPolicy, OutputNoDataPolicy>(
            std::forward<InputNoDataPolicy>(input_no_data_policy),
            std::forward<OutputNoDataPolicy>(output_no_data_policy))(
        values1, values2, result);
}

} // namespace algebra
} // namespace fern

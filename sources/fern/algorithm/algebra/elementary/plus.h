#pragma once
#include "fern/core/assert.h"
#include "fern/core/base_class.h"
#include "fern/core/type_traits.h"
#include "fern/algorithm/core/operation_traits.h"
#include "fern/algorithm/policy/discard_domain_errors.h"
#include "fern/algorithm/policy/discard_range_errors.h"
#include "fern/algorithm/policy/dont_mark_no_data.h"
#include "fern/algorithm/policy/skip_no_data.h"
#include "fern/algorithm/algebra/binary_operation.h"
#include "fern/algorithm/algebra/result_type.h"


namespace fern {
namespace plus {
namespace detail {
namespace dispatch {

template<
    class Values1,
    class Values2,
    class R,
    class A1NumberCategory,
    class A2NumberCategory>
struct within_range
{
};


template<
    class Values1,
    class Values2,
    class R>
struct within_range<
    Values1,
    Values2,
    R,
    unsigned_integer_tag,
    unsigned_integer_tag>
{
    inline static constexpr bool calculate(
        Values1 const& values1,
        Values2 const& /* values2 */,
        R const& result)
    {
        FERN_STATIC_ASSERT(std::is_same,
            typename fern::Result<Values1, Values2>::type, R)

        // unsigned + unsigned
        // Overflow if result is smaller than one of the operands.
        return !(result < values1);
    }
};


template<
    class Values1,
    class Values2,
    class R>
struct within_range<
    Values1,
    Values2,
    R,
    signed_integer_tag,
    signed_integer_tag>
{
    inline static constexpr bool calculate(
        Values1 const& values1,
        Values2 const& values2,
        R const& result)
    {
        FERN_STATIC_ASSERT(std::is_same,
            typename fern::Result<Values1, Values2>::type, R)

        // signed + signed
        // Overflow/underflow if sign of result is different.
        return values2 > 0 ? !(result < values1) : !(result > values1);
    }
};


template<
    class Values1,
    class Values2,
    class R>
struct within_range<
    Values1,
    Values2,
    R,
    unsigned_integer_tag,
    signed_integer_tag>
{
    inline static bool calculate(
        Values1 const& values1,
        Values2 const& values2,
        R const& result)
    {
        // unsigned + signed
        // Switch arguments and forward request.
        return within_range<Values2, Values1, R, signed_integer_tag,
            unsigned_integer_tag>::calculate(values2, values1, result);
    }
};


template<
    class Values1,
    class Values2,
    class R>
struct within_range<
    Values1,
    Values2,
    R,
    signed_integer_tag,
    unsigned_integer_tag>
{
    inline static constexpr bool calculate(
        Values1 const& values1,
        Values2 const& /* values2 */,
        R const& result)
    {
        FERN_STATIC_ASSERT(std::is_same, typename fern::Result<Values1, Values2>
            ::type, R)

        return values1 > 0 ? result >= values1 : result <= values1;
    }
};


template<
    class Values1,
    class Values2,
    class R>
struct within_range<
    Values1,
    Values2,
    R,
    integer_tag,
    integer_tag>
{
    inline static bool calculate(
        Values1 const& values1,
        Values2 const& values2,
        R const& result)
    {
        return within_range<Values1, Values2, R,
            typename TypeTraits<Values1>::number_category,
            typename TypeTraits<Values2>::number_category>::calculate(values1,
                values2, result);
    }
};


template<
    class Values1,
    class Values2,
    class R>
struct within_range<
    Values1,
    Values2,
    R,
    floating_point_tag,
    floating_point_tag>
{
    inline static constexpr bool calculate(
        Values1 const& /* values1 */,
        Values2 const& /* values2 */,
        R const& result)
    {
        FERN_STATIC_ASSERT(std::is_same,
            typename fern::Result<Values1, Values2>::type, R)

        return std::isfinite(result);
    }
};


template<
    class Values1,
    class Values2,
    class R>
struct within_range<
    Values1,
    Values2,
    R,
    integer_tag,
    floating_point_tag>
{
    inline static constexpr bool calculate(
        Values1 const& /* values1 */,
        Values2 const& /* values2 */,
        R const& result)
    {
        FERN_STATIC_ASSERT(std::is_same,
            typename fern::Result<Values1, Values2>::type, R)

        // integral + float
        return std::isfinite(result);
    }
};


template<
    class Values1,
    class Values2,
    class R>
struct within_range<
    Values1,
    Values2,
    R,
    floating_point_tag,
    integer_tag>
{
    inline static constexpr bool calculate(
        Values1 const& /* values1 */,
        Values2 const& /* values2 */,
        R const& result)
    {
        FERN_STATIC_ASSERT(std::is_same,
            typename fern::Result<Values1, Values2>::type, R)

        // float + integral
        return std::isfinite(result);
    }
};

} // namespace dispatch
} // namespace detail


// All values are within the domain of valid values for plus.
template<
    class Values1,
    class Values2>
using OutOfDomainPolicy = DiscardDomainErrors;


template<
    class Values1,
    class Values2=Values1>
class OutOfRangePolicy
{

    FERN_STATIC_ASSERT(std::is_arithmetic, Values1)
    FERN_STATIC_ASSERT(std::is_arithmetic, Values2)

public:

    template<
        class R>
    inline constexpr bool within_range(
        Values1 const& values1,
        Values2 const& values2,
        R const& result) const
    {
        FERN_STATIC_ASSERT(std::is_arithmetic, R)

        using values1_tag = typename base_class<
            typename TypeTraits<Values1>::number_category, integer_tag>::type;
        using values2_tag = typename base_class<
            typename TypeTraits<Values2>::number_category, integer_tag>::type;

        return detail::dispatch::within_range<Values1, Values2, R, values1_tag,
            values2_tag>::calculate(values1, values2, result);
    }

protected:

    OutOfRangePolicy()=default;

    ~OutOfRangePolicy()=default;

};


template<
    class Values1,
    class Values2>
struct Algorithm
{

    FERN_STATIC_ASSERT(std::is_arithmetic, Values1)
    FERN_STATIC_ASSERT(std::is_arithmetic, Values2)

    template<
        class R>
    inline void operator()(
        Values1 const& values1,
        Values2 const& values2,
        R& result) const
    {
        result = static_cast<R>(values1) + static_cast<R>(values2);
    }

};

} // namespace plus


namespace algebra {

template<
    class Values1,
    class Values2,
    class Result,
    class OutOfDomainPolicy=DiscardDomainErrors,
    class OutOfRangePolicy=DiscardRangeErrors,
    class InputNoDataPolicy=SkipNoData,
    class OutputNoDataPolicy=DontMarkNoData
>
class Plus
{

public:

    using category = local_operation_tag;
    using A1 = Values1;
    using A1Value = typename ArgumentTraits<A1>::value_type;
    using A2 = Values2;
    using A2Value = typename ArgumentTraits<A2>::value_type;
    using R = Result;
    using RValue = typename ArgumentTraits<R>::value_type;

    FERN_STATIC_ASSERT(std::is_arithmetic, A1Value)
    FERN_STATIC_ASSERT(std::is_arithmetic, A2Value)
    FERN_STATIC_ASSERT(std::is_arithmetic, RValue)
    FERN_STATIC_ASSERT(std::is_same, RValue,
        typename fern::Result<A1, A2>::type)

    /// //! Type of the result of the operation.
    /// typedef typename Result<A1, A2>::type R;

    /// typedef typename ArgumentTraits<A1>::value_type A1Value;

    /// typedef typename ArgumentTraits<A2>::value_type A2Value;

    Plus()
        : _algorithm(plus::Algorithm<A1Value, A2Value>())
    {
    }

    Plus(
        InputNoDataPolicy&& input_no_data_policy,  // Universal reference.
        OutputNoDataPolicy&& output_no_data_policy)  // Universal reference.
        : _algorithm(
            std::forward<InputNoDataPolicy>(input_no_data_policy),
            std::forward<OutputNoDataPolicy>(output_no_data_policy),
            plus::Algorithm<A1Value, A2Value>())
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

    detail::dispatch::BinaryOperation<A1, A2, R,
        OutOfDomainPolicy, OutOfRangePolicy, InputNoDataPolicy,
        OutputNoDataPolicy, plus::Algorithm<A1Value, A2Value>,
        typename ArgumentTraits<A1>::argument_category,
        typename ArgumentTraits<A2>::argument_category> _algorithm;

};


//! Calculate the result of adding \a values1 to \a values2 and put it in \a result.
/*!
*/
template<
    class Values1,
    class Values2,
    class Result,
    class OutOfDomainPolicy=DiscardDomainErrors,
    class OutOfRangePolicy=DiscardRangeErrors,
    class InputNoDataPolicy=SkipNoData,
    class OutputNoDataPolicy=DontMarkNoData
>
void plus(
    Values1 const& values1,
    Values2 const& values2,
    Result& result)
{
    Plus<Values1, Values2, Result, OutOfDomainPolicy, OutOfRangePolicy,
        InputNoDataPolicy, OutputNoDataPolicy>()(values1, values2, result);
}


//! Calculate the result of adding \a values1 to \a values2 and put it in \a result.
/*!
*/
template<
    class Values1,
    class Values2,
    class Result,
    class OutOfDomainPolicy=DiscardDomainErrors,
    class OutOfRangePolicy=DiscardRangeErrors,
    class InputNoDataPolicy=SkipNoData,
    class OutputNoDataPolicy=DontMarkNoData
>
void plus(
    InputNoDataPolicy&& input_no_data_policy,  // Universal reference.
    OutputNoDataPolicy&& output_no_data_policy,  // Universal reference.
    Values1 const& values1,
    Values2 const& values2,
    Result& result)
{
    Plus<Values1, Values2, Result, OutOfDomainPolicy, OutOfRangePolicy,
        InputNoDataPolicy, OutputNoDataPolicy>(
            std::forward<InputNoDataPolicy>(input_no_data_policy),
            std::forward<OutputNoDataPolicy>(output_no_data_policy))(
        values1, values2, result);
}

} // namespace algebra
} // namespace fern

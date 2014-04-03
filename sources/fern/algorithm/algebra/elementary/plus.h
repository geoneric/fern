#pragma once
#include "fern/core/assert.h"
#include "fern/core/base_class.h"
#include "fern/core/type_traits.h"
#include "fern/algorithm/core/operation_traits.h"
#include "fern/algorithm/policy/discard_domain_errors.h"
#include "fern/algorithm/policy/discard_range_errors.h"
#include "fern/algorithm/policy/dont_mark_no_data.h"
#include "fern/algorithm/algebra/binary_operation.h"
#include "fern/algorithm/algebra/result_type.h"


namespace fern {
namespace plus {
namespace detail {
namespace dispatch {

template<
    class A1,
    class A2,
    class R,
    class A1NumberCategory,
    class A2NumberCategory>
struct within_range
{
};


template<
    class A1,
    class A2,
    class R>
struct within_range<
    A1,
    A2,
    R,
    unsigned_integer_tag,
    unsigned_integer_tag>
{
    inline static constexpr bool calculate(
        A1 const& argument1,
        A2 const& /* argument2 */,
        R const& result)
    {
        FERN_STATIC_ASSERT(std::is_same,
            typename fern::Result<A1, A2>::type, R)

        // unsigned + unsigned
        // Overflow if result is smaller than one of the operands.
        return !(result < argument1);
    }
};


template<
    class A1,
    class A2,
    class R>
struct within_range<
    A1,
    A2,
    R,
    signed_integer_tag,
    signed_integer_tag>
{
    inline static constexpr bool calculate(
        A1 const& argument1,
        A2 const& argument2,
        R const& result)
    {
        FERN_STATIC_ASSERT(std::is_same,
            typename fern::Result<A1, A2>::type, R)

        // signed + signed
        // Overflow/underflow if sign of result is different.
        return argument2 > 0 ? !(result < argument1) : !(result > argument1);
    }
};


template<
    class A1,
    class A2,
    class R>
struct within_range<
    A1,
    A2,
    R,
    unsigned_integer_tag,
    signed_integer_tag>
{
    inline static bool calculate(
        A1 const& argument1,
        A2 const& argument2,
        R const& result)
    {
        // unsigned + signed
        // Switch arguments and forward request.
        return within_range<A2, A1, R, signed_integer_tag,
            unsigned_integer_tag>::calculate(argument2, argument1, result);
    }
};


template<
    class A1,
    class A2,
    class R>
struct within_range<
    A1,
    A2,
    R,
    signed_integer_tag,
    unsigned_integer_tag>
{
    inline static constexpr bool calculate(
        A1 const& argument1,
        A2 const& /* argument2 */,
        R const& result)
    {
        FERN_STATIC_ASSERT(std::is_same, typename fern::Result<A1, A2>::type, R)

        return argument1 > 0 ? result >= argument1 : result <= argument1;
    }
};


template<
    class A1,
    class A2,
    class R>
struct within_range<
    A1,
    A2,
    R,
    integer_tag,
    integer_tag>
{
    inline static bool calculate(
        A1 const& argument1,
        A2 const& argument2,
        R const& result)
    {
        return within_range<A1, A2, R, typename TypeTraits<A1>::number_category,
            typename TypeTraits<A2>::number_category>::calculate(argument1,
                argument2, result);
    }
};


template<
    class A1,
    class A2,
    class R>
struct within_range<
    A1,
    A2,
    R,
    floating_point_tag,
    floating_point_tag>
{
    inline static constexpr bool calculate(
        A1 const& /* argument1 */,
        A2 const& /* argument2 */,
        R const& result)
    {
        FERN_STATIC_ASSERT(std::is_same, typename fern::Result<A1, A2>::type, R)

        return std::isfinite(result);
    }
};


template<
    class A1,
    class A2,
    class R>
struct within_range<
    A1,
    A2,
    R,
    integer_tag,
    floating_point_tag>
{
    inline static constexpr bool calculate(
        A1 const& /* argument1 */,
        A2 const& /* argument2 */,
        R const& result)
    {
        FERN_STATIC_ASSERT(std::is_same, typename fern::Result<A1, A2>::type, R)

        // integral + float
        return std::isfinite(result);
    }
};


template<
    class A1,
    class A2,
    class R>
struct within_range<
    A1,
    A2,
    R,
    floating_point_tag,
    integer_tag>
{
    inline static constexpr bool calculate(
        A1 const& /* argument1 */,
        A2 const& /* argument2 */,
        R const& result)
    {
        FERN_STATIC_ASSERT(std::is_same, typename fern::Result<A1, A2>::type, R)

        // float + integral
        return std::isfinite(result);
    }
};

} // namespace dispatch
} // namespace detail


// All values are within the domain of valid values for plus.
template<
    class A1,
    class A2>
using OutOfDomainPolicy = DiscardDomainErrors<A1, A2>;


template<
    class A1,
    class A2>
class OutOfRangePolicy
{

    FERN_STATIC_ASSERT(std::is_arithmetic, A1)
    FERN_STATIC_ASSERT(std::is_arithmetic, A2)

public:

    template<
        class R>
    inline constexpr bool within_range(
        A1 const& argument1,
        A2 const& argument2,
        R const& result) const
    {
        FERN_STATIC_ASSERT(std::is_arithmetic, R)

        typedef typename base_class<typename TypeTraits<A1>::number_category,
            integer_tag>::type a1_tag;
        typedef typename base_class<typename TypeTraits<A2>::number_category,
            integer_tag>::type a2_tag;

        return detail::dispatch::within_range<A1, A2, R, a1_tag, a2_tag>::
            calculate(argument1, argument2, result);
    }

protected:

    OutOfRangePolicy()=default;

    ~OutOfRangePolicy()=default;

};


template<
    class A1,
    class A2>
struct Algorithm
{
    FERN_STATIC_ASSERT(std::is_arithmetic, A1)
    FERN_STATIC_ASSERT(std::is_arithmetic, A2)

    template<
        class R>
    inline void operator()(
        A1 const& argument1,
        A2 const& argument2,
        R& result) const
    {
        result = static_cast<R>(argument1) + static_cast<R>(argument2);
    }

};

} // namespace plus


namespace algebra {

//! Implementation of the plus operation, for two (collections of) arithmetic types.
/*!
  \tparam    A1 Type of first argument.
  \tparam    A2 Type of second argument.
  \tparam    OutOfDomainPolicy Policy class for handling out-of-domain values.
  \tparam    OutOfRangePolicy Policy class for handling out-of-range values.
  \tparam    NoDataPolicy Policy class for handling no-data values.
*/
template<
    class A1,
    class A2,
    class OutOfDomainPolicy=DiscardDomainErrors<
        typename ArgumentTraits<A1>::value_type,
        typename ArgumentTraits<A2>::value_type>,
    class OutOfRangePolicy=DiscardRangeErrors,
        /// <
        /// typename ArgumentTraits<A1>::value_type,
        /// typename ArgumentTraits<A2>::value_type>,
    class NoDataPolicy=DontMarkNoData
>
struct Plus
{

    typedef local_operation_tag category;

    //! Type of the result of the operation.
    typedef typename Result<A1, A2>::type R;

    typedef typename ArgumentTraits<A1>::value_type A1Value;

    typedef typename ArgumentTraits<A2>::value_type A2Value;

    Plus()
        : algorithm(plus::Algorithm<A1Value, A2Value>())
    {
    }

    Plus(
        NoDataPolicy&& no_data_policy)
        : algorithm(std::forward<NoDataPolicy>(no_data_policy),
            plus::Algorithm<A1Value, A2Value>())
    {
    }

    inline void operator()(
        A1 const& argument1,
        A2 const& argument2,
        R& result)
    {
        algorithm.calculate(argument1, argument2, result);
    }

    template<
        class Indices>
    inline void operator()(
        Indices const& indices,
        A1 const& argument1,
        A2 const& argument2,
        R& result)
    {
        algorithm.calculate(indices, argument1, argument2, result);
    }

    detail::dispatch::BinaryOperation<A1, A2, R,
        OutOfDomainPolicy, OutOfRangePolicy, NoDataPolicy,
        plus::Algorithm<
            typename ArgumentTraits<A1>::value_type,
            typename ArgumentTraits<A2>::value_type>,
        typename ArgumentTraits<A1>::argument_category,
        typename ArgumentTraits<A2>::argument_category> algorithm;

};


//! Calculate the result of adding \a argument1 to \a argument2 and put it in \a result.
/*!
  \tparam    A1 Type of \a argument1.
  \tparam    A2 Type of \a argument2.
  \param     argument1 First argument to add.
  \param     argument2 Second argument to add.
  \return    Result is stored in argument \a result.

  This function uses the Plus class template with default policies for handling
  out-of-domain values, out-of-range values and no-data.
*/
template<
    class A1,
    class A2>
void plus(
    A1 const& argument1,
    A2 const& argument2,
    typename Plus<A1, A2>::R& result)
{
    Plus<A1, A2>()(argument1, argument2, result);
}

} // namespace algebra
} // namespace fern

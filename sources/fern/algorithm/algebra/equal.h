#pragma once
#include "fern/algorithm/algebra/binary_operation.h"
#include "fern/algorithm/algebra/result_type.h"
#include "fern/algorithm/policy/dont_mark_no_data.h"
#include "fern/algorithm/policy/discard_domain_errors.h"
#include "fern/algorithm/policy/discard_range_errors.h"
#include "fern/core/argument_traits.h"
#include "fern/core/assert.h"


namespace fern {
namespace equal {

// All values are within the domain of valid values for equal.
template<
    class A1,
    class A2>
using OutOfDomainPolicy = DiscardDomainErrors<A1, A2>;


// All values are within the range of valid values for equal (given the bool
// output value type and the algorithm).
template<
    class A1,
    class A2>
using OutOfRangePolicy = DiscardRangeErrors; /// <A1, A2>;


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
        result = static_cast<uint8_t>(argument1 == argument2);
    }

};

} // namespace equal


namespace algebra {

//! Implementation of the equal operation, for two (collections of) arithmetic types.
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
        // <
        // typename ArgumentTraits<A1>::value_type,
        // typename ArgumentTraits<A2>::value_type>,
    class NoDataPolicy=DontMarkNoData
>
struct Equal
{

    //! Type of the result of the operation.
    typedef typename Result<A1, A2, bool>::type R;

    typedef typename ArgumentTraits<A1>::value_type A1Value;

    typedef typename ArgumentTraits<A2>::value_type A2Value;

    Equal()
        : algorithm(equal::Algorithm<A1Value, A2Value>())
    {
    }

    Equal(
        NoDataPolicy&& no_data_policy)
        : algorithm(std::forward<NoDataPolicy>(no_data_policy),
            equal::Algorithm<A1Value, A2Value>())
    {
    }

    inline void operator()(
        A1 const& argument1,
        A2 const& argument2,
        R& result)
    {
        algorithm.calculate(argument1, argument2, result);
    }

    detail::dispatch::BinaryOperation<A1, A2, R,
        OutOfDomainPolicy, OutOfRangePolicy, NoDataPolicy,
        equal::Algorithm<
            typename ArgumentTraits<A1>::value_type,
            typename ArgumentTraits<A2>::value_type>,
        typename ArgumentTraits<A1>::argument_category,
        typename ArgumentTraits<A2>::argument_category> algorithm;

};


//! Calculate the result of comparing \a argument1 to \a argument2 and put it in \a result.
/*!
  \tparam    A1 Type of \a argument1.
  \tparam    A2 Type of \a argument2.
  \param     argument1 First argument to compare.
  \param     argument2 Second argument to compare.
  \return    Result is stored in argument \a result.

  This function uses the Equal class template with default policies for handling
  out-of-domain values, out-of-range values and no-data.
*/
template<
    class A1,
    class A2>
void equal(
    A1 const& argument1,
    A2 const& argument2,
    typename Equal<A1, A2>::R& result)
{
    Equal<A1, A2>()(argument1, argument2, result);
}

} // namespace algebra
} // namespace fern

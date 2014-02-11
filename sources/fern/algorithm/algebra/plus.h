#pragma once
#include <algorithm>
#include <cmath>
#include <type_traits>
#include "fern/algorithm/algebra/result_type.h"
#include "fern/algorithm/policy/dont_mark_no_data.h"
#include "fern/algorithm/policy/discard_domain_errors.h"
#include "fern/algorithm/policy/discard_range_errors.h"
#include "fern/core/argument_traits.h"
#include "fern/core/base_class.h"
#include "fern/core/type_traits.h"


#define FERN_STATIC_ASSERT( \
        trait, \
        ...) \
    static_assert(trait<__VA_ARGS__>::value, "Assuming " #trait);


namespace fern {
namespace plus {
namespace detail {
namespace dispatch {

template<
    class A1,
    class A2,
    class R,
    class A1CollectionCategory,
    class A2CollectionCategory>
struct plus
{
};


template<
    class A1,
    class A2,
    class R>
struct plus<
    A1,
    A2,
    R,
    constant_tag,
    constant_tag>
{
    inline static constexpr void calculate(
        A1 const& argument1,
        A2 const& argument2,
        R& result)
    {
        result = static_cast<R>(argument1) + static_cast<R>(argument2);
    }
};


template<
    class A1,
    class A2,
    class R>
struct plus<
    A1,
    A2,
    R,
    collection_tag,
    collection_tag>
{
    inline static void calculate(
        A1 const& argument1,
        A2 const& argument2,
        R& result)
    {
        assert(argument1.size() == argument2.size());
        fern::resize(result, argument1);

        typename ArgumentTraits<A1>::const_iterator argument1_it =
            fern::begin(argument1);
        typename ArgumentTraits<A1>::const_iterator argument1_end =
            fern::end(argument1);
        typename ArgumentTraits<A2>::const_iterator argument2_it =
            fern::begin(argument2);
        typename ArgumentTraits<R>::iterator result_it = fern::begin(result);

        // typename A1::const_iterator argument1_it = argument1.begin();
        // typename A1::const_iterator argument1_end = argument1.end();
        // typename A2::const_iterator argument2_it = argument2.begin();
        // typename R::iterator result_it = result.begin();

        for(; argument1_it != argument1_end; ++argument1_it, ++argument2_it,
                ++result_it) {
            // *result_it = static_cast<typename R::value_type>(*argument1_it) +
            //     static_cast<typename R::value_type>(*argument2_it);
            *result_it =
                static_cast<typename ArgumentTraits<R>::value_type>(
                    *argument1_it) +
                static_cast<typename ArgumentTraits<R>::value_type>(
                    *argument2_it);
        }
    }
};


template<
    class A1,
    class A2,
    class R>
struct plus<
    A1,
    A2,
    R,
    constant_tag,
    collection_tag>
{
    inline static void calculate(
        A1 const& argument1,
        A2 const& argument2,
        R& result)
    {
        fern::resize(result, argument2);

        typename A2::const_iterator argument2_it = argument2.begin();
        typename A2::const_iterator argument2_end = argument2.end();
        typename R::iterator result_it = result.begin();

        for(; argument2_it != argument2_end; ++argument2_it, ++result_it) {
            *result_it = static_cast<typename R::value_type>(argument1) +
                static_cast<typename R::value_type>(*argument2_it);
        }
    }
};


template<
    class A1,
    class A2,
    class R>
struct plus<
    A1,
    A2,
    R,
    collection_tag,
    constant_tag>
{
    inline static void calculate(
        A1 const& argument1,
        A2 const& argument2,
        R& result)
    {
        plus<A2, A1, R, constant_tag, collection_tag>::calculate(argument2,
            argument1, result);
    }
};


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
            typename fern::result<A1, A2>::type, R)

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
            typename fern::result<A1, A2>::type, R)

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
        FERN_STATIC_ASSERT(std::is_same,
            typename fern::result<A1, A2>::type, R)

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
        FERN_STATIC_ASSERT(std::is_same, typename fern::result<A1, A2>::type, R)

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
        FERN_STATIC_ASSERT(std::is_same, typename fern::result<A1, A2>::type, R)

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
        FERN_STATIC_ASSERT(std::is_same, typename fern::result<A1, A2>::type, R)

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
using Domain = DiscardDomainErrors<A1, A2>;


template<
    class A1,
    class A2>
struct Range
{

    template<
        class R>
    inline constexpr bool within_range(
        A1 const& argument1,
        A2 const& argument2,
        R const& result) const
    {
        FERN_STATIC_ASSERT(std::is_arithmetic, A1)
        FERN_STATIC_ASSERT(std::is_arithmetic, A2)
        FERN_STATIC_ASSERT(std::is_arithmetic, R)

        typedef typename base_class<typename TypeTraits<A1>::number_category,
            integer_tag>::type a1_tag;
        typedef typename base_class<typename TypeTraits<A2>::number_category,
            integer_tag>::type a2_tag;

        return detail::dispatch::within_range<A1, A2, R, a1_tag, a2_tag>::
            calculate(argument1, argument2, result);
    }

protected:

    Range()=default;

    ~Range()=default;

};

} // namespace plus


//! Implementation of the plus operation, for two (collections of) arithmetic types.
/*!
  \tparam    A1 Type of first argument.
  \tparam    A2 Type of second argument.
*/
template<
    class A1,
    class A2,
    class OutOfDomainPolicy=DiscardDomainErrors<A1, A2>,
    class OutOfRangePolicy=DiscardRangeErrors<A1, A2>,
    class NoDataPolicy=DontMarkNoData
>
struct Plus:
    public OutOfDomainPolicy,
    public OutOfRangePolicy,
    public NoDataPolicy
{
    //! Type of the result of the operation.
    typedef typename result<A1, A2>::type R;

    //! Return the result of calculating \a argument1 plus \a argument2.
    /*!
      \param     argument1 First argument.
      \param     argument2 Second argument.
    */
    inline R operator()(
        A1 const& argument1,
        A2 const& argument2) const
    {
        R result;
        this->operator()(argument1, argument2, result);
        return result;
    }

    inline void operator()(
        A1 const& argument1,
        A2 const& argument2,
        R& result) const
    {
        // if(!OutOfDomainPolicy::within_domain(argument1, argument2)) {
        //     NoDataPolicy::set_no_data(...);
        // }
        // else {
        //     // calculate result.
        // }

        // if(!OutOfRangePolicy::within_range(argument1, argument2, result)) {
        //     NoDataPolicy::set_no_data(...);
        // }

        plus::detail::dispatch::plus<A1, A2, R,
            typename ArgumentTraits<A1>::argument_category,
            typename ArgumentTraits<A2>::argument_category>::calculate(
                argument1, argument2, result);
    }
};


namespace algebra {

//! Calculate the result of adding \a argument1 to \a argument2 and put it in \a result.
/*!
  \tparam    A1 Type of \a argument1.
  \tparam    A2 Type of \a argument2.
  \param     argument1 First argument to add.
  \param     argument2 Second argument to add.
  \return    Result is stored in argument \a result.
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


//! Calculate the result of adding \a argument1 to \a argument2 and return the result.
/*!
  \tparam    A1 Type of \a argument1.
  \tparam    A2 Type of \a argument2.
  \param     argument1 First argument to add.
  \param     argument2 Second argument to add.
  \return    Result of adding \a argument1 to \a argument2.
*/
template<
    class A1,
    class A2>
typename Plus<A1, A2>::R plus(
    A1 const& argument1,
    A2 const& argument2)
{
    typename Plus<A1, A2>::R result;
    Plus<A1, A2>()(argument1, argument2, result);
    return result;
}

} // namespace algebra
} // namespace fern









// //! Implementation of the plus operation, for two ranges of arithmetic types.
// /*!
//   \tparam    A1 Type of first argument.
//   \tparam    A2 Type of second argument.
// */
// template<
//     template<class ValueType> class Range1,
//     template<class ValueType> class Range2,
//     class = typename std::enable_if<
//         !std::is_arithmetic<typename Range1::value_type>::value &&
//         !std::is_arithmetic<typename Range2::value_type>::value>::type
// >
// struct Plus
// {
//     //! Type of a single value in the first argument range.
//     typedef Range::value_type A1;
// 
//     //! Type of a single value in the second argument range.
//     typedef Range::value_type A2;
// 
//     //! Type of the result of the operation.
//     typedef Range<decltype(A1() + A2())> R;
// 
//     //! Return the result of calculating \a argument1 plus \a argument2.
//     /*!
//       \param     argument1 First argument.
//       \param     argument2 Second argument.
//     */
//     inline R operator()(
//         Range<A1> const& argument1,
//         Range<A2> const& argument2) const
//     {
//         assert(argument1.size() == argument2.size());
//         R result(argument1.size());
// 
//         for(size_t i = 0; i < result.size(); ++i) {
//             result[i] = argument1[i] + argument2[i];
//         }
//     }
// };


// template<
//     class A1,
//     class A2,
//     >
// R plus(
//     A1 const& argument1,
//     A2 const& argument2)
// {
// 
// }




    // ,
    // class ThreadingModel=SingleThreaded,
    // class DomainPolicy=DiscardDomainErrors,
    // class RangePolicy=DiscardRangeErrors>
    // public ThreadingModel,
    // public DomainPolicy,
    // public RangePolicy


// DomainPolicy specific for algorithm.
// policy.within_domain(argument);


// Free functions, templated to the argument type.
// set_value(result_value, value);
// set_no_data(result_value);

// template<
//     class Result,
//     class Value>
// void               set_value           (Result& result,
//                                         Value const& value);
// 
// template<
//     class Result,
//     class Value>
// void               set_value           (Result& result,
//                                         Value const& value,
//                                         size_t i);
// 
// template<
//     class Result>
// void               set_no_data         (Result& result);
// 
// 
// template<
//     class Result,
//     class Value>
// inline void set_value(
//     Result& result,
//     Value const& value)
// {
//     // Default case, for when Result and Value are arithmetic.
//     result = value;
// }
// 
// 
// template<
//     class Result>
// void set_no_data(
//     Result& result)
// {
//     // Default case, for when Result and Value are arithmetic.
//     result = TypeTraits<Result>::no_data_value;
// }
// 
// 
// template<
//     class Value>
//     MaskedArrayValue<Value, 1>
//     class Result>
// void set_no_data<MaskedArrayValue<Valuej(
//     Result& result)
// {
//     // Default case, for when Result and Value are arithmetic.
//     result = TypeTraits<Result>::no_data_value;
// }










// // Constants.
// if(!within_domain(argument1) || !within_domain(argument2)) {
//     set_as_no_data(result_value);
// }
// else if(!within_range(argument1, argument2)) {
//     set_as_no_data(result_value);
// }
// else {
//     set_value(result, calculate(argument1, argument2));
// }


// Integers:
// - signed + signed: overflow if sign of result is different.
// - signed + unsigned: will never overflow.


// // Arrays.
// for(size_t i = 0; i < size; ++i) {
//     A1 const& value1(argument1[i]);
//     A2 const& value2(argument2[i]);
// 
//     // Domain check for binary operations take all arguments into account.
//     // Some operations need to have a look at both arguments at the same
//     // time.
//     if(!within_domain(value1, value2)) {
//         // Domain is not OK. Don't set a value and mark the value as no-data.
//         set_as_no_data(result, i);
//     }
//     else {
//         // Domain is OK. Calculate and set the value.
//         value = calculate(value1, value2);
//         set_value(result, value, i);
//
//         // Range check takes all arguments and the result into account.
//         if(!within_range(value1, value2, result)) {
//             // Overflow or underflow. Mark the value as no-data.
//             set_as_no_data(result, i);
//         }
//     }
// }


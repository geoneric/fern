#pragma once
#include <algorithm>
#include <cmath>
#include <type_traits>
#include <boost/mpl/max_element.hpp>
#include <boost/mpl/placeholders.hpp>
#include <boost/mpl/sizeof.hpp>
#include <boost/mpl/transform_view.hpp>
#include <boost/mpl/vector.hpp>
#include "fern/core/type_traits.h"
#include "fern/core/typelist.h"



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

#define FERN_STATIC_ASSERT( \
        trait, \
        ...) \
    static_assert(trait<__VA_ARGS__>::value, "Assuming " #trait);


namespace fern {
namespace plus {
namespace detail {
namespace mpl = boost::mpl;

template<
    class A1,
    class A2,
    class A1NumberCategory,
    class A2NumberCategory>
struct ArgumentTypeTraitsDispatch
{
};


template<
    class A1,
    class A2>
struct ArgumentTypeTraitsDispatch<
    A1,
    A2,
    unsigned_integer_tag,
    unsigned_integer_tag>
{
    // Pick the largest type.
    typedef mpl::vector<A1, A2> types;
    typedef typename mpl::max_element<mpl::transform_view<types,
        mpl::sizeof_<mpl::_1>>>::type iter;
    typedef typename mpl::deref<typename iter::base>::type R;
};


template<
    class A1,
    class A2>
struct ArgumentTypeTraitsDispatch<
    A1,
    A2,
    signed_integer_tag, signed_integer_tag>
{
    // Pick the largest type.
    typedef mpl::vector<A1, A2> types;
    typedef typename mpl::max_element<mpl::transform_view<types,
        mpl::sizeof_<mpl::_1>>>::type iter;
    typedef typename mpl::deref<typename iter::base>::type R;
};


template<
    class T1,
    class T2>
constexpr auto min(
    T1 const& value1,
    T2 const& value2) -> decltype(value1 < value2 ? value1 : value2)
{
    return value1 < value2 ? value1 : value2;
}


template<
    class A1,
    class A2>
struct ArgumentTypeTraitsDispatch<
    A1,
    A2,
    unsigned_integer_tag,
    signed_integer_tag>
{
    typedef Typelist<uint8_t, uint16_t, uint32_t, uint64_t> UnsignedTypes;
    typedef Typelist<int8_t, int16_t, int32_t, int64_t> SignedTypes;

    // Find index of A1 in list of unsigned types. Determine type of next
    // larger type in list of signed types. -> Type1
    typedef typename at<min(find<A1, UnsignedTypes>::value + 1,
        size<SignedTypes>::value - 1), SignedTypes>::type Type1;

    // Result type is largest_type(Type1, A2)
    typedef typename mpl::if_c<(sizeof(Type1) > sizeof(A2)), Type1, A2>::type R;
};


template<
    class A1,
    class A2>
struct ArgumentTypeTraitsDispatch<
    A1,
    A2,
    signed_integer_tag,
    unsigned_integer_tag>
{
    // Switch template arguments.
    typedef typename ArgumentTypeTraitsDispatch<A2, A1, unsigned_integer_tag,
        signed_integer_tag>::R R;
};


template<
    class A1,
    class A2>
struct ArgumentTypeTraitsDispatch<
    A1,
    A2,
    floating_point_tag,
    floating_point_tag>
{
    // Pick the largest type.
    typedef mpl::vector<A1, A2> types;
    typedef typename mpl::max_element<mpl::transform_view<types,
        mpl::sizeof_<mpl::_1>>>::type iter;
    typedef typename mpl::deref<typename iter::base>::type R;
};


template<
    class A1,
    class A2,
    bool is_same>
struct ArgumentTypeTraits
{
};


template<
    class A1,
    class A2>
struct ArgumentTypeTraits<A1, A2, true>
{
    // R == A1 == A2.
    typedef A1 R;
};


template<
    class A1,
    class A2>
struct ArgumentTypeTraits<A1, A2, false>
{
    // // C++ conversion rules.
    // typedef decltype(A1() + A2()) R;

    typedef typename ArgumentTypeTraitsDispatch<A1, A2,
        typename TypeTraits<A1>::number_category,
        typename TypeTraits<A2>::number_category>::R R;
};


template<
    class A1,
    class A2,
    class R>
inline constexpr bool within_integral_range(
    A1 const& argument1,
    A2 const& argument2,
    R const& result,
    std::true_type const /* is_signed */,
    std::true_type const /* is_signed */)
{
    FERN_STATIC_ASSERT(std::is_integral, A1)
    FERN_STATIC_ASSERT(std::is_integral, A2)
    FERN_STATIC_ASSERT(std::is_integral, R)
    FERN_STATIC_ASSERT(std::is_signed, A1)
    FERN_STATIC_ASSERT(std::is_signed, A2)
    FERN_STATIC_ASSERT(std::is_signed, R)

    // signed + signed
    // Overflow if sign of result is different.
    return argument2 > 0 ? !(result < argument1) : !(result > argument1);
}


template<
    class A1,
    class A2,
    class R>
inline constexpr bool within_integral_range(
    A1 const& argument1,
    A2 const& /* argument2 */,
    R const& result,
    std::false_type const /* is_signed */,
    std::false_type const /* is_signed */)
{
    FERN_STATIC_ASSERT(std::is_integral, A1)
    FERN_STATIC_ASSERT(std::is_integral, A2)
    FERN_STATIC_ASSERT(std::is_integral, R)
    FERN_STATIC_ASSERT(std::is_unsigned, A1)
    FERN_STATIC_ASSERT(std::is_unsigned, A2)
    FERN_STATIC_ASSERT(std::is_unsigned, R)

    // unsigned + unsigned
    // Overflow if result is smaller than one of the operands.
    return !(result < argument1);
}


template<
    class A1,
    class A2,
    class R>
inline constexpr bool within_integral_range(
    A1 const& argument1,
    A2 const& argument2,
    R const& result,
    std::true_type const /* is_signed */,
    std::false_type const /* is_signed */)
{
    FERN_STATIC_ASSERT(std::is_integral, A1)
    FERN_STATIC_ASSERT(std::is_integral, A2)
    FERN_STATIC_ASSERT(std::is_integral, R)
    FERN_STATIC_ASSERT(std::is_signed, A1)
    FERN_STATIC_ASSERT(std::is_unsigned, A2)
    FERN_STATIC_ASSERT(std::is_signed, R)

    // signed + unsigned
    // C++ will apply type promotion. R has this resulting type.
    return within_integral_range<R, R, R>(R(argument1), R(argument2), result,
        std::is_signed<R>(), std::is_signed<R>());
}


template<
    class A1,
    class A2,
    class R>
inline constexpr bool within_integral_range(
    A1 const& argument1,
    A2 const& argument2,
    R const& result,
    std::false_type const is_signed1,
    std::true_type const is_signed2)
{
    FERN_STATIC_ASSERT(std::is_integral, A1)
    FERN_STATIC_ASSERT(std::is_integral, A2)
    FERN_STATIC_ASSERT(std::is_integral, R)
    FERN_STATIC_ASSERT(std::is_unsigned, A1)
    FERN_STATIC_ASSERT(std::is_signed, A2)
    FERN_STATIC_ASSERT(std::is_signed, R)

    // unsigned + signed
    // Switch arguments and forward request.
    return within_integral_range(argument2, argument1, result, is_signed2,
        is_signed1);
}


template<
    class A1,
    class A2,
    class R>
inline constexpr bool within_range(
    A1 const& argument1,
    A2 const& argument2,
    R const& result,
    std::true_type const /* is_integral */,
    std::true_type const /* is_integral */)
{
    FERN_STATIC_ASSERT(std::is_integral, A1)
    FERN_STATIC_ASSERT(std::is_integral, A2)
    FERN_STATIC_ASSERT(std::is_integral, R)

    return within_integral_range<A1, A2, R>(argument1, argument2, result,
        std::is_signed<A1>(), std::is_signed<A2>());
}


template<
    class A1,
    class A2,
    class R>
inline constexpr bool within_floating_point_range(
    A1 const& /* argument1 */,
    A2 const& /* argument2 */,
    R const& result)
{
    FERN_STATIC_ASSERT(std::is_floating_point, A1)
    FERN_STATIC_ASSERT(std::is_floating_point, A2)
    FERN_STATIC_ASSERT(std::is_floating_point, R)

    // float + float
    return std::isfinite(result);
}


template<
    class A1,
    class A2,
    class R>
inline constexpr bool within_range(
    A1 const& argument1,
    A2 const& argument2,
    R const& result,
    std::false_type const /* is_integral */,
    std::false_type const /* is_integral */)
{
    FERN_STATIC_ASSERT(std::is_floating_point, A1)
    FERN_STATIC_ASSERT(std::is_floating_point, A2)
    FERN_STATIC_ASSERT(std::is_floating_point, R)

    return within_floating_point_range<A1, A2, R>(argument1, argument2, result);
}


template<
    class A1,
    class A2,
    class R>
inline constexpr bool within_range(
    A1 const& argument1,
    A2 const& argument2,
    R const& result,
    std::true_type const /* is_integral */,
    std::false_type const /* is_integral */)
{
    FERN_STATIC_ASSERT(std::is_integral, A1)
    FERN_STATIC_ASSERT(std::is_floating_point, A2)
    FERN_STATIC_ASSERT(std::is_floating_point, R)
    FERN_STATIC_ASSERT(std::is_same, A2, R);

    // integral + float
    // C++ will apply type promotion. R has this resulting type.
    return within_range<R, A2, R>(R(argument1), argument2, result,
        std::is_integral<R>(), std::is_integral<R>());
}


template<
    class A1,
    class A2,
    class R>
inline constexpr bool within_range(
    A1 const& argument1,
    A2 const& argument2,
    R const& result,
    std::false_type const is_integral1,
    std::true_type const is_integral2)
{
    FERN_STATIC_ASSERT(std::is_floating_point, A1)
    FERN_STATIC_ASSERT(std::is_integral, A2)
    FERN_STATIC_ASSERT(std::is_floating_point, R)
    FERN_STATIC_ASSERT(std::is_same, A1, R);

    // float + integral
    // Switch arguments and forward request.
    return within_range(argument2, argument1, result, is_integral2,
        is_integral1);
}

} // namespace detail


template<
    class A1,
    class A2>
struct ArgumentTypeTraits
{
    typedef typename detail::ArgumentTypeTraits<A1, A2,
        std::is_same<A1, A2>::value>::R R;
};


struct Domain
{
    template<
        class A1,
        class A2>
    inline constexpr bool within_domain(
        A1 const& /* argument1 */,
        A2 const& /* argument2 */) const
    {
        // All values are within the domain of valid values for plus.
        return true;
    }
};


struct Range
{
    template<
        class A1,
        class A2,
        class R>
    inline bool within_range(
        A1 const& argument1,
        A2 const& argument2,
        R const& result) const
    {
        FERN_STATIC_ASSERT(std::is_arithmetic, A1)
        FERN_STATIC_ASSERT(std::is_arithmetic, A2)
        FERN_STATIC_ASSERT(std::is_arithmetic, R)

        return detail::within_range<A1, A2, R>(argument1, argument2,
            result, std::is_integral<A1>(), std::is_integral<A2>());
    }
};

} // namespace plus


//! Implementation of the plus operation, for two arithmetic types.
/*!
  \tparam    A1_ Type of first argument.
  \tparam    A2_ Type of second argument.
*/
template<
    class A1_,
    class A2_,
    class = typename std::enable_if<
        std::is_arithmetic<A1_>::value &&
        std::is_arithmetic<A2_>::value>::type
>
struct Plus
{
    //! Type of first argument value.
    typedef A1_ A1;

    //! Type of a second argument value.
    typedef A2_ A2;

    //! Type of the result of the operation.
    typedef typename plus::ArgumentTypeTraits<A1, A2>::R R;

    //! Return the result of calculating \a argument1 plus \a argument2.
    /*!
      \param     argument1 First argument.
      \param     argument2 Second argument.
    */
    inline constexpr R operator()(
        A1 const& argument1,
        A2 const& argument2) const
    {
        return argument1 + argument2;
    }
};

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



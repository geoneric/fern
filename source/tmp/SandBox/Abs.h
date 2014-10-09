#ifndef INCLUDED_CMATH
#include <cmath>
#define INCLUDED_CMATH
#endif

#ifndef INCLUDED_BOOST_STATIC_ASSERT
#include <boost/static_assert.hpp>
#define INCLUDED_BOOST_STATIC_ASSERT
#endif

#ifndef INCLUDED_BOOST_TYPE_TRAITS_IS_ARITHMETIC
#include <boost/type_traits/is_arithmetic.hpp>
#define INCLUDED_BOOST_TYPE_TRAITS_IS_ARITHMETIC
#endif

#ifndef INCLUDED_BOOST_TYPE_TRAITS_IS_FLOATING_POINT
#include <boost/type_traits/is_floating_point.hpp>
#define INCLUDED_BOOST_TYPE_TRAITS_IS_FLOATING_POINT
#endif

#ifndef INCLUDED_BOOST_TYPE_TRAITS_IS_SAME
#include <boost/type_traits/is_same.hpp>
#define INCLUDED_BOOST_TYPE_TRAITS_IS_SAME
#endif



template<typename T, bool isFloatingPointType>
struct absImplementation;



template<typename T>
struct absImplementation<T, false>
{
  static inline T calculate(T value)
  {
    return std::abs(value);
  }
};



template<typename T>
struct absImplementation<T, true>
{
  static inline T calculate(T value)
  {
    return std::fabs(value);
  }
};



template<typename Argument, typename Result>
inline Result abs(
         Argument value)
{
  BOOST_STATIC_ASSERT(boost::is_arithmetic<Argument>::value);
  // KDJ: For now. I can think of unsigned int = abs(int), but won't test that
  // KDJ: now.
  BOOST_STATIC_ASSERT((boost::is_same<Argument, Result>::value));

  return absImplementation<Argument,
              boost::is_floating_point<Argument>::value>::calculate(value);
}


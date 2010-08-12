#ifndef INCLUDED_CMATH
#include <cmath>
#define INCLUDED_CMATH
#endif

#ifndef INCLUDED_BOOST_STATIC_ASSERT
#include <boost/static_assert.hpp>
#define INCLUDED_BOOST_STATIC_ASSERT
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
struct sqrt_implementation;



template<typename T>
struct sqrt_implementation<T, true>
{
  static inline T calculate(T value)
  {
    return std::sqrt(value);
  }
};



template<typename Argument, typename Result>
inline Result sqrt(
         Argument value)
{
  BOOST_STATIC_ASSERT(boost::is_floating_point<Argument>::value);
  BOOST_STATIC_ASSERT((boost::is_same<Argument, Result>::value));

  return sqrt_implementation<Argument,
              boost::is_floating_point<Argument>::value>::calculate(value);
}


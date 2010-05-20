#ifndef INCLUDED_RANALLY_OPERATIONS_LOCAL_BINARY_PLUS
#define INCLUDED_RANALLY_OPERATIONS_LOCAL_BINARY_PLUS

#ifndef INCLUDED_BOOST_STATIC_ASSERT
#include <boost/static_assert.hpp>
#define INCLUDED_BOOST_STATIC_ASSERT
#endif

#ifndef INCLUDED_BOOST_TYPE_TRAITS
#include <boost/type_traits.hpp>
#define INCLUDED_BOOST_TYPE_TRAITS
#endif

#ifndef INCLUDED_RANALLY_OPERATIONS_POLICIES_DOMAIN
#include "Ranally/Operations/Policies/Domain.h"
#define INCLUDED_RANALLY_OPERATIONS_POLICIES_DOMAIN
#endif



namespace ranally {
namespace operations {
namespace binary {
namespace plus {

template<typename T>
class Algorithm
{
  inline static T operator()(
         T argument1,
         T argument2)
  {
    BOOST_STATIC_ASSERT(boost::is_arithmetic<T>::value);

    return argument1 + argument2;
  }
};

template<typename T>
class DomainPolicy: public ranally::operations::Policies::DummyDomain<T>
{
};

namespace detail {

template<typename T, bool isFloatingPoint>
class RangePolicy
{
  static bool      inRange             (T argument1,
                                        T argument2,
                                        T result);
};

template<typename T>
class RangePolicy<T, true>
{
  inline static bool inRange(
         T argument1,
         T argument2,
         T result)
  {
    // All floating points values are valid.
    return true;
  }
};

template<typename T, bool isSigned>
class RangePolicyForIntegrals
{
  bool             inRange             (T argument1,
                                        T argument2,
                                        T result);
};

template<typename T>
class RangePolicyForIntegrals<T, true>
{
  inline static bool inRange(
         T argument1,
         T argument2,
         T result)
  {
    // Signed integrals.
    if(argument1 < T(0) && argument2 < T(0)) {
      return result < T(0);
    }
    else if(argument1 > T(0) && argument2 > T(0)) {
      return result > T(0);
    }
    else {
      return true;
    }
  }
};

template<typename T>
class RangePolicyForIntegrals<T, false>
{
  inline static bool inRange(
         T argument1,
         T argument2,
         T result)
  {
    // Unsigned integrals.
    return !(result < argument1);
  }
};

template<typename T>
class RangePolicy<T, false>: public RangePolicyForIntergrals<T,
         boost::is_signed<Argument>::value>
{
};

} // namespace detail

template<typename T>
class RangePolicy: public detail::RangePolicy<T,
         boost::is_floating_point<T>::value>
{
};

} // namespace plus
} // namespace binary
} // namespace operations
} // namespace ranally

#endif

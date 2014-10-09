#ifndef INCLUDED_RANALLY_OPERATIONS_LOCAL_UNARY_PLUS
#define INCLUDED_RANALLY_OPERATIONS_LOCAL_UNARY_PLUS

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

#ifndef INCLUDED_RANALLY_OPERATIONS_POLICIES_RANGE
#include "Ranally/Operations/Policies/Range.h"
#define INCLUDED_RANALLY_OPERATIONS_POLICIES_RANGE
#endif



namespace ranally {
namespace operations {
namespace unary {
namespace plus {

template<typename T>
struct Algorithm
{
  inline static T calculate(
         T argument)
  {
    BOOST_STATIC_ASSERT(boost::is_arithmetic<T>::value);

    // No-op.
    return argument;
  }
};

template<typename T>
struct DomainPolicy: public ranally::operations::policies::DummyDomain<T>
{
};

template<typename T>
struct RangePolicy: public ranally::operations::policies::DummyRange<T>
{
};

} // namespace plus
} // namespace unary
} // namespace operations
} // namespace ranally

#endif

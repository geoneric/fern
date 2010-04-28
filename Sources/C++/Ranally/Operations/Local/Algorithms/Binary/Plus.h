#ifndef INCLUDED_RANALLY_OPERATIONS_LOCAL_ALGORITHMS_BINARY_PLUS
#define INCLUDED_RANALLY_OPERATIONS_LOCAL_ALGORITHMS_BINARY_PLUS

#ifndef INCLUDED_RANALLY_OPERATIONS_POLICIES_DOMAIN
#include "Ranally/Operations/Policies/Domain.h"
#define INCLUDED_RANALLY_OPERATIONS_POLICIES_DOMAIN
#endif



namespace Ranally {
namespace Algorithms {
namespace Plus {

template<typename T>
class Algorithm {
  static inline T operator()(
         T argument1,
         T argument2)
  {
    return argument1 + argument2;
  }
};

class DomainPolicy: public Ranally::Operations::Policies::DummyDomain
{
};

template<typename T>
class RangePolicy
{
  static inline bool inRange(
         T argument1,
         T argument2,
         T result)
  {
    // TODO Return true if T is floating point.
    // TODO Return !(result < argument1) if T is unsigned.
    // TODO For signed int:
    //   if(argument1 < cast(T)0 && argument2 < cast(T)0) {
    //     return argument1 + argument2 < cast(T)0;
    //   }
    //   else if(argument1 > cast(T)0 && argument2 > cast(T)0) {
    //     return argument1 + argument2 >= cast(T)0;
    //   }
    //     else {
    //       return true;
    //     }
    //   }
    // }

    return false;
  }
};

} // namespace Plus
} // namespace Algorithms
} // namespace Ranally

#endif

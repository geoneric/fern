#ifndef INCLUDED_RANALLY_POINTPTR
#define INCLUDED_RANALLY_POINTPTR

#include <boost/shared_ptr.hpp>

#define DEFINE_SHARED_POINTER_TYPE(                                            \
  className)                                                                   \
  class className;                                                             \
  typedef boost::shared_ptr<className> className##Ptr;

namespace ranally {

DEFINE_SHARED_POINTER_TYPE(PointAttribute)
DEFINE_SHARED_POINTER_TYPE(PointDomain)
DEFINE_SHARED_POINTER_TYPE(PointFeature)
DEFINE_SHARED_POINTER_TYPE(PointValue)

} // namespace ranally

#undef DEFINE_SHARED_POINTER_TYPE

#endif

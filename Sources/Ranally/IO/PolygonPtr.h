#ifndef INCLUDED_RANALLY_POLYGONPTR
#define INCLUDED_RANALLY_POLYGONPTR

#include <boost/shared_ptr.hpp>

#define DEFINE_SHARED_POINTER_TYPE(                                            \
  className)                                                                   \
  class className;                                                             \
  typedef boost::shared_ptr<className> className##Ptr;

namespace ranally {

DEFINE_SHARED_POINTER_TYPE(PolygonAttribute)
DEFINE_SHARED_POINTER_TYPE(PolygonDomain)
DEFINE_SHARED_POINTER_TYPE(PolygonFeature)
DEFINE_SHARED_POINTER_TYPE(PolygonValue)

} // namespace ranally

#undef DEFINE_SHARED_POINTER_TYPE

#endif

#pragma once
#include <memory>
#include <vector>


namespace ranally {

#define DEFINE_SHARED_POINTER_TYPE(                                            \
    className)                                                                 \
    class className;                                                           \
    typedef std::shared_ptr<className> className##Ptr;

DEFINE_SHARED_POINTER_TYPE(PolygonAttribute)
DEFINE_SHARED_POINTER_TYPE(PolygonDomain)
DEFINE_SHARED_POINTER_TYPE(PolygonFeature)
DEFINE_SHARED_POINTER_TYPE(PolygonValue)

#undef DEFINE_SHARED_POINTER_TYPE

typedef std::vector<PolygonAttributePtr> PolygonAttributes;
typedef std::vector<PolygonValuePtr> PolygonValues;

} // namespace ranally

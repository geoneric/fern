#pragma once
#include <memory>
#include <vector>


namespace geoneric {

#define DEFINE_SHARED_POINTER_TYPE(                                            \
    className)                                                                 \
    class className;                                                           \
    typedef std::shared_ptr<className> className##Ptr;

DEFINE_SHARED_POINTER_TYPE(PointAttribute)
DEFINE_SHARED_POINTER_TYPE(PointDomain)
DEFINE_SHARED_POINTER_TYPE(PointFeature)
DEFINE_SHARED_POINTER_TYPE(PointValue)

#undef DEFINE_SHARED_POINTER_TYPE

typedef std::vector<PointAttributePtr> PointAttributes;
typedef std::vector<PointValuePtr> PointValues;

} // namespace geoneric

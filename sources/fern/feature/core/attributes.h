#pragma once
#include <memory>
#include "fern/feature/core/array_value.h"
#include "fern/feature/core/constant_attribute.h"
#include "fern/feature/core/box.h"
#include "fern/feature/core/feature.h"
#include "fern/feature/core/masked_array_value.h"
#include "fern/feature/core/point.h"
#include "fern/feature/core/spatial_attribute.h"
#include "fern/feature/core/spatial_domain.h"

namespace fern {
namespace d1 {

template<
    class T>
using ArrayValue = ArrayValue<T, 1>;

template<
    class T>
using ArrayValuePtr = std::shared_ptr<ArrayValue<T>>;

template<
    class T>
using MaskedArrayValue = MaskedArrayValue<T, 1>;

template<
    class T>
using MaskedArrayValuePtr = std::shared_ptr<MaskedArrayValue<T>>;

typedef Point<double, 1> Point;
typedef SpatialDomain<Point> PointDomain;

}


namespace d2 {

template<
    class T>
using ArrayValue = ArrayValue<T, 2>;

template<
    class T>
using ArrayValuePtr = std::shared_ptr<ArrayValue<T>>;

template<
    class T>
using MaskedArrayValue = MaskedArrayValue<T, 2>;

template<
    class T>
using MaskedArrayValuePtr = std::shared_ptr<MaskedArrayValue<T>>;

typedef Point<double, 2> Point;
typedef Box<Point> Box;
typedef SpatialDomain<Box> BoxDomain;
typedef SpatialDomain<Point> PointDomain;

} // namespace d2


namespace d3 {

template<
    class T>
using ArrayValue = ArrayValue<T, 3>;

template<
    class T>
using ArrayValuePtr = std::shared_ptr<ArrayValue<T>>;

template<
    class T>
using MaskedArrayValue = MaskedArrayValue<T, 3>;

template<
    class T>
using MaskedArrayValuePtr = std::shared_ptr<MaskedArrayValue<T>>;

typedef Point<double, 3> Point;
typedef Box<Point> Box;
typedef SpatialDomain<Box> BoxDomain;
typedef SpatialDomain<Point> PointDomain;

} // namespace d3


typedef SpatialDomain<d2::Box> FieldDomain;

template<
    class T>
using FieldValue = d2::MaskedArrayValue<T>;

template<
    class T>
using FieldValuePtr = std::shared_ptr<FieldValue<T>>;

template<
    class T>
using FieldAttribute = SpatialAttribute<FieldDomain, FieldValuePtr<T>>;

template<
    class T>
using FieldAttributePtr = std::shared_ptr<FieldAttribute<T>>;

} // namespace fern

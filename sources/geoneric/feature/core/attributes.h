#pragma once
#include <memory>
#include "geoneric/feature/core/array_value.h"
#include "geoneric/feature/core/constant_attribute.h"
#include "geoneric/feature/core/box.h"
#include "geoneric/feature/core/feature.h"
#include "geoneric/feature/core/point.h"
#include "geoneric/feature/core/spatial_attribute.h"
#include "geoneric/feature/core/spatial_domain.h"

namespace geoneric {
namespace d1 {

template<
    class T>
using ArrayValue = ArrayValue<T, 1>;

template<
    class T>
using ArrayValuePtr = std::shared_ptr<ArrayValue<T>>;

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

typedef Point<double, 3> Point;
typedef Box<Point> Box;
typedef SpatialDomain<Box> BoxDomain;
typedef SpatialDomain<Point> PointDomain;

} // namespace d3


//!
/*!
  \tparam    .
  \param     .
  \return    .
  \exception .
  \warning   .
  \sa        .

  The default value stored per box is a 1D array of T values.
*/
template<
    class T,
    class V=d1::ArrayValuePtr<T>>
using FieldAttribute = SpatialAttribute<d2::BoxDomain, V>;

template<
    class T>
using FieldAttributePtr = std::shared_ptr<FieldAttribute<T>>;

} // namespace geoneric

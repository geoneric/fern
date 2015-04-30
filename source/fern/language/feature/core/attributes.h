// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include <memory>
#include "fern/feature/core/point.h"
#include "fern/language/feature/core/array_value.h"
#include "fern/language/feature/core/constant_attribute.h"
#include "fern/language/feature/core/box.h"
#include "fern/language/feature/core/feature.h"
#include "fern/language/feature/core/masked_array_value.h"
#include "fern/language/feature/core/spatial_attribute.h"
#include "fern/language/feature/core/spatial_domain.h"

namespace fern {
namespace language {
namespace d1 {

template<
    typename T>
using ArrayValue = ArrayValue<T, 1>;

template<
    typename T>
using ArrayValuePtr = std::shared_ptr<ArrayValue<T>>;

template<
    typename T>
using MaskedArrayValue = MaskedArrayValue<T, 1>;

template<
    typename T>
using MaskedArrayValuePtr = std::shared_ptr<MaskedArrayValue<T>>;

using Point = Point<double, 1>;
using PointDomain = SpatialDomain<Point>;

}


namespace d2 {

template<
    typename T>
using ArrayValue = ArrayValue<T, 2>;

template<
    typename T>
using ArrayValuePtr = std::shared_ptr<ArrayValue<T>>;

template<
    typename T>
using MaskedArrayValue = MaskedArrayValue<T, 2>;

template<
    typename T>
using MaskedArrayValuePtr = std::shared_ptr<MaskedArrayValue<T>>;

using Point = Point<double, 2>;
using Box = Box<Point>;
using BoxDomain = SpatialDomain<Box>;
using PointDomain = SpatialDomain<Point>;

} // namespace d2


namespace d3 {

template<
    typename T>
using ArrayValue = ArrayValue<T, 3>;

template<
    typename T>
using ArrayValuePtr = std::shared_ptr<ArrayValue<T>>;

template<
    typename T>
using MaskedArrayValue = MaskedArrayValue<T, 3>;

template<
    typename T>
using MaskedArrayValuePtr = std::shared_ptr<MaskedArrayValue<T>>;

using Point = Point<double, 3>;
using Box = Box<Point>;
using BoxDomain = SpatialDomain<Box>;
using PointDomain = SpatialDomain<Point>;

} // namespace d3


using FieldDomain = SpatialDomain<d2::Box>;

template<
    typename T>
using FieldValue = d2::MaskedArrayValue<T>;

template<
    typename T>
using FieldValuePtr = std::shared_ptr<FieldValue<T>>;

template<
    typename T>
using FieldAttribute = SpatialAttribute<FieldDomain, FieldValuePtr<T>>;

template<
    typename T>
using FieldAttributePtr = std::shared_ptr<FieldAttribute<T>>;

} // namespace fern
} // namespace language

// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include <boost/geometry/core/access.hpp>
#include <boost/geometry/core/cs.hpp>
#include <boost/geometry/geometries/point.hpp>


namespace fern {

template<
    typename CoordinateType,
    size_t nr_dimensions,
    typename CoordinateSystem=boost::geometry::cs::cartesian>
using Point = boost::geometry::model::point<CoordinateType, nr_dimensions,
    CoordinateSystem>;

using boost::geometry::set;
using boost::geometry::get;

} // namespace fern

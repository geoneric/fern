#pragma once

#include <boost/geometry/core/access.hpp>
#include <boost/geometry/core/cs.hpp>
#include <boost/geometry/geometries/point.hpp>


namespace geoneric {

template<
    class CoordinateType,
    size_t nr_dimensions,
    class CoordinateSystem=boost::geometry::cs::cartesian>
using Point = boost::geometry::model::point<CoordinateType, nr_dimensions,
    CoordinateSystem>;

using boost::geometry::set;
using boost::geometry::get;

} // namespace geoneric

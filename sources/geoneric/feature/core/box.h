#pragma once
#include <boost/geometry/geometries/box.hpp>


namespace geoneric {

template<
    class Point>
using Box = boost::geometry::model::box<Point>;

} // namespace geoneric

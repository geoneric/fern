#pragma once
#include <boost/geometry/geometries/box.hpp>


namespace fern {

template<
    class Point>
using Box = boost::geometry::model::box<Point>;

} // namespace fern

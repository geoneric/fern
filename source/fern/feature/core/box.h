#pragma once
#include <boost/geometry/geometries/box.hpp>


namespace fern {

template<
    typename Point>
using Box = boost::geometry::model::box<Point>;

} // namespace fern

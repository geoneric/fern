#ifndef INCLUDED_RANALLY_GEOMETRY
#define INCLUDED_RANALLY_GEOMETRY

#include <vector>
#include <boost/geometry/core/cs.hpp>
#include <boost/geometry/geometries/point.hpp>
#include <boost/geometry/geometries/polygon.hpp>
#include <boost/shared_ptr.hpp>



namespace ranally {

// TODO What about the coordinate system? How to go about that? Can we pick
//      one and project all incoming data to that cs?

typedef double Coordinate;

typedef boost::geometry::model::point<Coordinate, 2,
  boost::geometry::cs::cartesian > Point;

namespace detail {
  //! Points are stored in clock-wise direction.
  bool const clockWise = true;

  //! Polygons are closed (last point == first point).
  bool const closed = true;
} // namespace detail

typedef boost::geometry::model::polygon<Point, detail::clockWise,
  detail::closed> Polygon;

typedef std::vector<Point> Points;

typedef std::vector<Polygon> Polygons;

typedef boost::shared_ptr<Points> PointsPtr;

typedef boost::shared_ptr<Polygons> PolygonsPtr;

} // namespace ranally

#endif

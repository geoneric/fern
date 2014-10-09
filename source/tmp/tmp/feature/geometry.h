#pragma once
// #include <memory>
// #include <vector>
#include <boost/geometry/core/access.hpp>
#include <boost/geometry/core/cs.hpp>
#include <boost/geometry/geometries/point.hpp>
#include <boost/geometry/geometries/polygon.hpp>


namespace fern {

// TODO What about the coordinate system? How to go about that? Can we pick
//      one and project all incoming data to that cs?

typedef double Coordinate;
typedef int64_t Fid;


namespace detail {

  typedef boost::geometry::model::point<Coordinate, 2,
      boost::geometry::cs::cartesian> PointBase;
}

typedef detail::PointBase Point;


// class Point:
//     public detail::PointBase
// {
// 
// public:
// 
//                    Point               ();
// 
//                    Point               (Coordinate x,
//                                         Coordinate y);
//                                         // Fid fid);
// 
// private:
// 
//     // //! Feature-id.
//     // Fid            _fid;
// 
// };
// 
// 
// inline Point::Point()
// 
//     : detail::PointBase()
// 
// {
// }
// 
// 
// inline Point::Point(
//     Coordinate x,
//     Coordinate y)
//     // Fid fid)
// 
//     : detail::PointBase(x, y) // ,
//       // _fid(fid)
// 
// {
// }



namespace detail {
    //! Points are stored in clock-wise direction.
    bool const clockWise = true;

    //! Polygons are open (last point != first point).
    bool const closed = false;
} // namespace detail

typedef boost::geometry::model::polygon<Point, detail::clockWise,
    detail::closed> Polygon;

// typedef std::vector<Point> Points;

// typedef std::vector<Polygon> Polygons;

// typedef std::shared_ptr<Points> PointsPtr;

// typedef std::shared_ptr<Polygons> PolygonsPtr;

} // namespace fern


namespace boost {
namespace geometry {
namespace model {

inline bool operator==(
    fern::Point const& lhs,
    fern::Point const& rhs)
{
   return
       boost::geometry::get<0>(lhs) == boost::geometry::get<0>(rhs) &&
       boost::geometry::get<1>(lhs) == boost::geometry::get<1>(rhs)
       ;
}

} // namespace model
} // namespace geometry

namespace test_tools {

inline std::ostream& operator<<(
    std::ostream& stream,
    fern::Point const& point)
{
    stream << "(" <<
        boost::geometry::get<0>(point)
        << ", " <<
        boost::geometry::get<1>(point)
        << ")"
        ;
    return stream;
}

} // namespace test_tools
} // namespace boost


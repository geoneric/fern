#define BOOST_TEST_MODULE fern feature
#include <boost/test/unit_test.hpp>
#include "fern/feature/core/box.h"
#include "fern/feature/core/point.h"
#include "fern/feature/core/spatial_domain.h"


template<
    class CoordinateType,
    size_t nr_dimensions,
    class CoordinateSystem>
inline bool operator==(
    fern::Point<CoordinateType, nr_dimensions,
        CoordinateSystem> const& lhs,
    fern::Point<CoordinateType, nr_dimensions,
        CoordinateSystem> const& rhs)
{
    return
        fern::get<0>(lhs) == fern::get<0>(rhs) &&
        fern::get<1>(lhs) == fern::get<1>(rhs);
}


template<
    class Point>
inline bool operator==(
    fern::Box<Point> const& lhs,
    fern::Box<Point> const& rhs)
{
    return
        lhs.min_corner() == rhs.min_corner() &&
        lhs.max_corner() == rhs.max_corner();
}


BOOST_AUTO_TEST_SUITE(spatial_domain)

BOOST_AUTO_TEST_CASE(spatial_domain)
{
    typedef fern::Point<double, 2> Point;
    typedef fern::Box<Point> Box;
    typedef fern::SpatialDomain<Box> BoxDomain;

    BoxDomain spatial_domain;

    Point south_west;
    Point north_east;
    fern::set<0>(south_west, 1.1);
    fern::set<1>(south_west, 2.2);
    fern::set<0>(north_east, 3.3);
    fern::set<1>(north_east, 4.4);
    Box box1(south_west, north_east);
    Box box2(south_west, north_east);

    // Whatever the geometry, adding one to a domain should result in a
    // unique geometry id.
    BoxDomain::GID gid1 = spatial_domain.add(box1);
    BoxDomain::GID gid2 = spatial_domain.add(box2);
    BOOST_CHECK_NE(gid1, gid2);

    // Test geometries themselves.
    BOOST_CHECK(spatial_domain.geometry(gid1) == box1);
    BOOST_CHECK(spatial_domain.geometry(gid2) == box2);
}

BOOST_AUTO_TEST_SUITE_END()
